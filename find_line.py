#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_line.py -- 雲に出る「細い白い直線」を、帯の平均ではなく線として探す。

  これまでの道具(find_wedge.py)は緯度帯や経度列の平均を見ていたので、
  細い斜め線は必ず平均に埋もれて検出できなかった。
  ここでは「周囲より明るい細い筋」を直接叩き、直線状に連なるものだけ残す。

  python3 find_line.py --faces 20260823_2137
  python3 find_line.py --eq clouds_src.png
  python3 find_line.py --faces 20260823_2137 --eq clouds_src.png --dump

判定:
  equirect に無く 面にある      → render_face が作っている（継ぎ目/サンプリング）
  両方にある                    → make_clouds.py の合成が作っている
  面の境界(端0-2px)に沿っている → キューブ面の継ぎ目で確定
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

FACE_SIZE = 1024
TILE = 256
GRID = 4


def boxblur(a, r):
    if r < 1:
        return a.astype(np.float32)
    out = a.astype(np.float64)
    for _ in range(2):
        for ax in (0, 1):
            n = out.shape[ax]
            pad = np.pad(out, [(r + 1, r) if i == ax else (0, 0) for i in range(2)], mode="edge")
            c = np.cumsum(pad, axis=ax)
            hi = np.take(c, np.arange(2 * r + 1, n + 2 * r + 1), axis=ax)
            lo = np.take(c, np.arange(0, n), axis=ax)
            out = (hi - lo) / (2 * r + 1)
    return out.astype(np.float32)


def load_faces(date_dir):
    """DDS 96枚(6面 x 4x4タイル)を 6枚の 1024x1024 に戻す。"""
    import quicktex.dds

    faces = []
    for f in range(6):
        img = np.zeros((FACE_SIZE, FACE_SIZE), np.uint8)
        for row in range(GRID):
            for col in range(GRID):
                p = os.path.join(date_dir, "%d_2_%d_%d.dds" % (f, col, row))
                if not os.path.exists(p):
                    raise SystemExit("見つからない: " + p)
                d = quicktex.dds.read(p).decode()
                # quicktex のバージョンで API が違うので順に試す
                for conv in ("to_image", "to_pil", "image"):
                    if hasattr(d, conv):
                        o = getattr(d, conv)
                        t = o() if callable(o) else o
                        break
                else:
                    t = d
                if not isinstance(t, np.ndarray):
                    t = np.asarray(t.convert("L") if hasattr(t, "convert") else t)
                if t.ndim == 3:
                    t = t[:, :, 0]
                a = t
                img[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE] = a
        faces.append(img)
    return faces


def ridge(img, width=1, ctx=6):
    """細い明るい筋の強さ。幅 width 程度の筋だけが大きくなる。

    近傍(width)の平均から、やや広い近傍(ctx)の平均を引く＝バンドパス。
    雲の塊は ctx でも平均が上がるので消え、細い筋だけ残る。
    """
    f = img.astype(np.float32)
    near = boxblur(f, width)
    far = boxblur(f, ctx)
    r = near - far
    return r


def report(img, name, thr_sigma=6.0, minlen=40, edge_px=3):
    r = ridge(img)
    s = float(r.std())
    z = r / max(s, 1e-6)
    hot = z > thr_sigma
    n = int(hot.sum())
    print("  %-14s ridge sd %.2f  z>%.0f の画素 %d (%.4f%%)"
          % (name, s, thr_sigma, n, 100.0 * n / hot.size))
    if n == 0:
        return None
    ys, xs = np.nonzero(hot)
    H, W = img.shape
    # 面の端に寄っているか（継ぎ目の判定）
    d = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
    near_edge = float((d <= edge_px).mean())
    # 直線性: 主成分の細長さ
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], 1)
    pts -= pts.mean(0)
    if len(pts) >= 3:
        w_, v_ = np.linalg.eigh(np.cov(pts.T))
        elong = float(np.sqrt(max(w_[1], 1e-9) / max(w_[0], 1e-9)))
        ang = float(np.degrees(np.arctan2(v_[1, 1], v_[0, 1])) % 180.0)
    else:
        elong, ang = 0.0, 0.0
    print("     位置 x %d-%d / y %d-%d   端(<=%dpx)に %.0f%%   細長さ %.1f   角度 %.0f度"
          % (xs.min(), xs.max(), ys.min(), ys.max(), edge_px, 100 * near_edge, elong, ang))
    if near_edge > 0.5:
        print("     ★端に集中＝キューブ面の継ぎ目")
    elif elong > 6:
        print("     ★細長い＝直線状の人工物")
    return dict(z=z, hot=hot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", metavar="DATE_DIR")
    ap.add_argument("--eq", metavar="PNG")
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--dump", action="store_true", help="検出結果をPNGに出す")
    a = ap.parse_args()
    if not (a.faces or a.eq):
        ap.print_help()
        return

    if a.eq:
        print("equirect: %s" % a.eq)
        im = np.asarray(Image.open(a.eq).convert("L"))
        res = report(im, "equirect", a.sigma)
        if res and a.dump:
            Image.fromarray((res["hot"] * 255).astype(np.uint8)).save("line_eq.png")
            print("     -> line_eq.png")

    if a.faces:
        print("faces: %s" % a.faces)
        for i, img in enumerate(load_faces(a.faces)):
            res = report(img, "face %d" % i, a.sigma)
            if res and a.dump:
                Image.fromarray((res["hot"] * 255).astype(np.uint8)).save("line_face%d.png" % i)
        if a.dump:
            print("     -> line_face*.png")


if __name__ == "__main__":
    main()
