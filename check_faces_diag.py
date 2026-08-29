#!/usr/bin/env python3
"""check_faces_diag.py -- キューブ面に入った「斜めの直線」を検出する。

check_seams.py が行と列しか見ないので、斜めの線は平均に埋もれて素通りする。
こちらは 0〜179 度の全方向にラドン積分をかけて、どの向きの直線でも拾う。

  python3 check_faces_diag.py --tiles 20260827_1414
  python3 check_faces_diag.py --tiles 20260828_0026 --tiles 20260827_1414   (並べて比較)
  python3 check_faces_diag.py --selftest        (合成画像で検出器の動作確認)

判定: 最大z が NG_Z 以上 かつ 角度別中央値の NG_RATIO 倍以上 なら NG。
終了コード 1 で NG。
"""
import argparse
import os
import sys
import numpy as np

try:
    from scipy import ndimage
except ImportError:
    raise SystemExit("pip3 install scipy numpy")

FACE_NAMES = ["px", "nx", "py", "ny", "pz", "nz"]
NG_Z = 7.0
NG_RATIO = 2.3  # 2026-08-29 ラベル調整: クロ20260827_1414=2.82 / シロ5日分の最大=2.15 / selftest振幅3=2.40 が上限


# ---------- BC1 (DXT1) デコード ----------

def decode_bc1(data, w, h):
    """BC1 圧縮データを (h, w, 3) uint8 に展開する。"""
    bw, bh = w // 4, h // 4
    b = np.frombuffer(data[: bw * bh * 8], dtype=np.uint8).reshape(bh, bw, 8)
    c0 = b[..., 0].astype(np.uint16) | (b[..., 1].astype(np.uint16) << 8)
    c1 = b[..., 2].astype(np.uint16) | (b[..., 3].astype(np.uint16) << 8)

    def rgb565(c):
        r = ((c >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0)
        g = ((c >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0)
        bl = (c & 0x1F).astype(np.float32) * (255.0 / 31.0)
        return np.stack([r, g, bl], -1)

    p0, p1 = rgb565(c0), rgb565(c1)
    wide = (c0 > c1)[..., None]
    p2 = np.where(wide, (2 * p0 + p1) / 3.0, (p0 + p1) / 2.0)
    p3 = np.where(wide, (p0 + 2 * p1) / 3.0, 0.0)
    pal = np.stack([p0, p1, p2, p3], axis=2)          # (bh, bw, 4, 3)

    idx_bytes = b[..., 4:8]                            # 行ごとに1バイト
    sh = np.array([0, 2, 4, 6], dtype=np.uint8)
    idx = (idx_bytes[..., None] >> sh) & 0x03          # (bh, bw, 4行, 4列)

    out = np.take_along_axis(
        pal[:, :, None, None, :, :],
        idx[..., None, None].astype(np.intp),
        axis=4,
    )[..., 0, :]                                       # (bh, bw, 4, 4, 3)
    out = out.transpose(0, 2, 1, 3, 4).reshape(h, w, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def read_dds(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != b"DDS ":
        raise ValueError("DDS ではない: %s" % path)
    h = int.from_bytes(raw[12:16], "little")
    w = int.from_bytes(raw[16:20], "little")
    fourcc = raw[84:88]
    if fourcc != b"DXT1":
        raise ValueError("DXT1 以外は未対応 (%s): %s" % (fourcc, path))
    return decode_bc1(raw[128:], w, h)


def load_face(tiledir, face, grid=4):
    rows = []
    for r in range(grid):
        cols = []
        for c in range(grid):
            p = os.path.join(tiledir, "%d_2_%d_%d.dds" % (face, c, r))
            if not os.path.exists(p):
                raise SystemExit("見つからない: %s" % p)
            cols.append(read_dds(p))
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


# ---------- 直線検出 ----------

def luma(img):
    a = img.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def line_scan(L, step=2.0, bigsig=60.0, clip=25.0, detrend=151, widths=(5, 15, 41)):
    """全方位ラドンで最大zと角度別中央値を返す。

    幅の広い淡い帯を狙うので、画像側の高域は sigma=60 の粗い引き算だけにして
    帯を残し、細かい判定は積分後の1次元プロファイルを detrend してから行う。
    (画像側で細かい高域を取ると、幅40〜60pxの帯は低周波として消えてしまう)
    """
    hp = np.clip(L - ndimage.gaussian_filter(L, bigsig), -clip, clip)
    H = hp.shape[0]
    per_angle = []
    best = (0.0, 0.0, 0, 0)
    for ang in np.arange(0.0, 180.0, step):
        r = ndimage.rotate(hp, ang, reshape=False, order=1, cval=0.0)
        cut = int(H * 0.16)                      # 回転で欠ける端を捨てる
        prof = r[cut:H - cut].mean(0)
        lo, hi = cut, len(prof) - cut
        top, cand = 0.0, None
        for w in widths:
            p = ndimage.uniform_filter1d(prof, w)
            p = p - ndimage.uniform_filter1d(p, detrend)
            pv = p[lo:hi]
            med = np.median(pv)
            mad = np.median(np.abs(pv - med)) * 1.4826 + 1e-9
            z = (pv - med) / mad
            i = int(np.argmax(np.abs(z)))
            if abs(z[i]) > top:
                top = abs(z[i])
                cand = (abs(z[i]), ang, lo + i - len(prof) // 2, w)
        per_angle.append(top)
        if cand and cand[0] > best[0]:
            best = cand
    return best, float(np.median(per_angle))


def check_dir(tiledir, step, quiet=False):
    print("[面の斜め検査] %s" % tiledir)
    worst = 0.0
    ng = False
    for f, name in enumerate(FACE_NAMES):
        face = load_face(tiledir, f)
        (z, ang, off, w), med = line_scan(luma(face), step=step)
        ratio = z / max(med, 1e-6)
        bad = (z >= NG_Z and ratio >= NG_RATIO)
        ng = ng or bad
        worst = max(worst, ratio)
        print("    face%d (%s): 最大z=%.1f 角度%5.1f度 中心から%+5d 帯幅%2d "
              "中央値%.1f 比%.2f  %s"
              % (f, name, z, ang, off, w, med, ratio, "NG" if bad else "OK"))
    print("---- 判定 ----")
    print("NG" if ng else "OK")
    return ng


def selftest():
    rng = np.random.default_rng(0)
    base = ndimage.gaussian_filter(rng.normal(128, 40, (1024, 1024)), 6)
    yy, xx = np.mgrid[0:1024, 0:1024]
    band = np.exp(-((xx - 0.23 * yy - 300) ** 2) / (2 * 25.0 ** 2))
    res = []
    for amp, tag in ((0, "線なし    "), (3, "振幅3/255 "), (6, "振幅6/255 "), (12, "振幅12/255")):
        (z, ang, off, w), med = line_scan(base + band * amp)
        r = z / max(med, 1e-6)
        bad = (z >= NG_Z and r >= NG_RATIO)
        res.append(bad)
        print("  %s 最大z=%5.1f 角度%5.1f 中心から%+5d 帯幅%2d 中央値%.1f 比%.2f  %s"
              % (tag, z, ang, off, w, med, r, "NG" if bad else "OK"))
    ok = (not res[0]) and all(res[1:])
    print("検出器は" + ("正常" if ok else "**要調整**"))
    return not ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", action="append", default=[],
                    help="96枚の .dds が入ったディレクトリ(複数可)")
    ap.add_argument("--step", type=float, default=2.0, help="角度刻み(度)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(1 if selftest() else 0)
    if not args.tiles:
        ap.error("--tiles か --selftest を指定")
    ng = False
    for d in args.tiles:
        ng = check_dir(d, args.step) or ng
        print()
    sys.exit(1 if ng else 0)


if __name__ == "__main__":
    main()
