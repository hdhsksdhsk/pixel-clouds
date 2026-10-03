#!/usr/bin/env python3
"""cloud_match.py
今の雲(equirect, clouds_src.png)を、内蔵の雲(clouds.ktx)の緯度帯ごとの明るさ分布に寄せる。
暗くする方向だけ (out = min(元, 変換後))。晴れの所に雲は足さない。
緯度帯は5度ごと、分位は101点、緯度方向は隣の帯と線形補間。

使い方:
  初回(内蔵の分位をJSONに保存):
    python3 cloud_match.py 入力.png 出力.png --ref-strip ~/Desktop/clouds_builtin_strip.png
  2回目以降(JSONを使う):
    python3 cloud_match.py 入力.png 出力.png
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image

BAND = 5.0
NB = int(round(180 / BAND))
CENTERS = -90.0 + BAND * (np.arange(NB) + 0.5)
LEVELS = np.linspace(0.0, 100.0, 101)
GROUPS = [("北60-90", 60, 90), ("北40-60", 40, 60), ("赤道付近", -10, 10), ("南40-60", -60, -40)]


def wquant(v, w, levels=LEVELS):
    o = np.argsort(v, kind="stable")
    v = v[o].astype(np.float64)
    wo = w[o].astype(np.float64)
    c = np.cumsum(wo)
    c = (c - 0.5 * wo) / c[-1] * 100.0
    return np.interp(levels, c, v)


def to_gray(a):
    if a.ndim == 3:
        return a[..., :3].astype(np.float64).mean(axis=2)
    return a.astype(np.float64)


def builtin_samples(strip_path):
    g = to_gray(np.asarray(Image.open(strip_path)))
    N = g.shape[1]
    if g.shape[0] != 6 * N:
        sys.exit(f"中止: 内蔵の雲の縦並びPNGの形が {g.shape[1]}x{g.shape[0]}（期待 {N}x{6 * N}）")
    idx = (np.arange(N) + 0.5) / N * 2.0 - 1.0
    s, t = np.meshgrid(idx, idx)
    one = np.ones_like(s)
    faces = [(one, -t, -s), (-one, -t, s), (s, one, t), (s, -one, -t), (s, -t, one), (-s, -t, -one)]
    w = (1.0 + s * s + t * t) ** -1.5
    vals, lats, wts = [], [], []
    for f, (x, y, z) in enumerate(faces):
        lat = np.degrees(np.arcsin(z / np.sqrt(x * x + y * y + z * z)))
        vals.append(g[f * N:(f + 1) * N].ravel())
        lats.append(lat.ravel())
        wts.append(w.ravel())
    return np.concatenate(vals), np.concatenate(lats), np.concatenate(wts)


def eq_samples(g):
    H, W = g.shape
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    L = np.repeat(lat[:, None], W, axis=1)
    return g.ravel(), L.ravel(), np.cos(np.radians(L)).ravel()


def band_quantiles(vals, lats, wts):
    b = np.clip(((lats + 90.0) / BAND).astype(int), 0, NB - 1)
    q = np.zeros((NB, LEVELS.size))
    for i in range(NB):
        m = b == i
        if not m.any():
            sys.exit(f"中止: 緯度帯 {i} が空")
        q[i] = wquant(vals[m], wts[m])
    return q


def group_medians(vals, lats, wts):
    out = {}
    for name, lo, hi in GROUPS:
        m = (lats >= lo) & (lats < hi)
        out[name] = float(wquant(vals[m], wts[m], np.array([50.0]))[0])
    return out


def apply_match(g, qs, qr):
    H, _ = g.shape
    eps = np.arange(LEVELS.size) * 1e-6
    out = np.empty_like(g)
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    for r in range(H):
        p = (lat[r] - CENTERS[0]) / BAND
        b0 = int(np.clip(np.floor(p), 0, NB - 2))
        w = float(np.clip(p - b0, 0.0, 1.0))
        row = g[r]
        m0 = np.interp(row, qs[b0] + eps, qr[b0])
        m1 = np.interp(row, qs[b0 + 1] + eps, qr[b0 + 1])
        out[r] = np.minimum(row, (1.0 - w) * m0 + w * m1)
    return out


def fmt(d):
    return "  ".join(f"{k} {v:6.1f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description="今の雲を内蔵の雲の緯度帯分布に寄せる（暗くする方向だけ）")
    ap.add_argument("src", help="入力 equirect (clouds_src.png)")
    ap.add_argument("out", help="出力 PNG")
    ap.add_argument("--ref-strip", help="内蔵の雲の縦並びPNG。指定すると分位を計算して --ref-json に保存")
    ap.add_argument("--ref-json", default="builtin_quantiles.json", help="内蔵の雲の分位JSON")
    ap.add_argument("--preview", help="上=元 / 下=変換後 の見比べ画像を書き出す")
    a = ap.parse_args()

    if a.ref_strip:
        bv, bl, bw = builtin_samples(a.ref_strip)
        qr = band_quantiles(bv, bl, bw)
        gm = group_medians(bv, bl, bw)
        with open(a.ref_json, "w") as f:
            json.dump({"band_deg": BAND, "levels": LEVELS.size, "q": qr.tolist(), "group_medians": gm}, f)
        print(f"内蔵の分位を保存: {a.ref_json}")
    else:
        try:
            with open(a.ref_json) as f:
                d = json.load(f)
        except FileNotFoundError:
            sys.exit(f"中止: {a.ref_json} が無い。初回は --ref-strip を付ける")
        if d.get("band_deg") != BAND or d.get("levels") != LEVELS.size:
            sys.exit("中止: JSONの帯幅か分位点数がスクリプトと合わない")
        qr = np.array(d["q"])
        gm = d["group_medians"]
    print("内蔵   中央値: " + fmt(gm))

    raw = np.asarray(Image.open(a.src))
    g = to_gray(raw)
    if g.shape[1] != 2 * g.shape[0]:
        sys.exit(f"中止: 入力が equirect(横=縦x2)ではない: {g.shape[1]}x{g.shape[0]}")
    sv, sl, sw = eq_samples(g)
    qs = band_quantiles(sv, sl, sw)
    print("変換前 中央値: " + fmt(group_medians(sv, sl, sw)))

    out = apply_match(g, qs, qr)
    ov, ol, ow = eq_samples(out)
    print("変換後 中央値: " + fmt(group_medians(ov, ol, ow)))

    o8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if raw.ndim == 3:
        res = raw.copy()
        for c in range(min(3, raw.shape[2])):
            res[..., c] = o8
        Image.fromarray(res).save(a.out)
    else:
        Image.fromarray(o8).save(a.out)
    print(f"書き出し: {a.out}")

    if a.preview:
        top = np.clip(np.rint(g), 0, 255).astype(np.uint8)
        pv = Image.fromarray(np.vstack([top, np.full((8, g.shape[1]), 255, np.uint8), o8]))
        pv = pv.resize((1024, int(pv.height * 1024 / pv.width)), Image.LANCZOS)
        pv.save(a.preview)
        print(f"見比べ: {a.preview}")


if __name__ == "__main__":
    main()
