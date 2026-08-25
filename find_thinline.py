#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_thinline.py

equirect の雲素材 (clouds_src.png など) から「細くて薄い直線」を探す。
平均や大域しきい値では細線は埋もれるので、方向を振ってラドン変換で
「長く一直線に続くか」を z 値で出す。

使い方:
    python3 find_thinline.py                       clouds_src.png の日本の東を見る
    python3 find_thinline.py <png>                 別の画像を見る
    python3 find_thinline.py <png> 120 170 -15 55  経度min max 緯度min max を指定

出力の読み方:
    対象域の z と、比較用の対照域の z を並べて出す。
    対照域と同じくらい (だいたい 4 以下) なら直線なし。
    対照域より 1.5 倍以上большой なら人工的な直線がある。
"""

import sys
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from skimage.transform import rotate


def load(path):
    im = Image.open(path).convert("L")
    return np.asarray(im).astype(np.float32), im.size


def crop(a, W, H, lon0, lon1, lat0, lat1):
    x0 = int((lon0 + 180.0) / 360.0 * W)
    x1 = int((lon1 + 180.0) / 360.0 * W)
    y0 = int((90.0 - lat1) / 180.0 * H)
    y1 = int((90.0 - lat0) / 180.0 * H)
    return a[y0:y1, x0:x1], (x0, y0)


def line_score(sub, clip=6.0):
    """細線だけ残してラドン。返り値 (z, 角度, 断面の位置)"""
    hp = sub - median_filter(sub, size=(5, 5))
    sig = np.clip(hp, -clip, clip)
    ok = np.ones_like(sig, dtype=np.float32)
    best = (0.0, 0.0, 0)
    for ang in np.arange(-75, 76, 1.0):
        R = rotate(sig, ang, resize=True, order=1, preserve_range=True)
        C = rotate(ok, ang, resize=True, order=1, preserve_range=True)
        cs = C.sum(0)
        m = cs > 0.55 * cs.max()
        if m.sum() < 20:
            continue
        prof = R.sum(0)[m] / cs[m]
        med = np.median(prof)
        mad = 1.4826 * np.median(np.abs(prof - med)) + 1e-9
        z = np.abs((prof - med) / mad)
        k = int(np.argmax(z))
        if z[k] > best[0]:
            best = (float(z[k]), float(ang), k)
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "clouds_src.png"
    if len(sys.argv) >= 6:
        lon0, lon1, lat0, lat1 = [float(v) for v in sys.argv[2:6]]
    else:
        lon0, lon1, lat0, lat1 = 120.0, 170.0, -15.0, 55.0

    a, (W, H) = load(path)
    print("画像 %s  %dx%d  経度 %.2f 度/px" % (path, W, H, 360.0 / W))
    print()

    sub, _ = crop(a, W, H, lon0, lon1, lat0, lat1)
    z, ang, k = line_score(sub)
    print("【対象】経度 %.0f〜%.0f / 緯度 %.0f〜%.0f  (%dx%d px)"
          % (lon0, lon1, lat0, lat1, sub.shape[1], sub.shape[0]))
    print("   直線スコア z = %.2f   角度 %.0f 度" % (z, ang))
    print()

    print("【対照】同じ大きさで別の海域を3つ")
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    ctrl = [(-150.0, -150.0 + dlon, lat0, lat1),
            (-40.0, -40.0 + dlon, lat0, lat1),
            (60.0, 60.0 + dlon, -60.0, -60.0 + dlat)]
    zs = []
    for c in ctrl:
        s, _ = crop(a, W, H, *c)
        if s.size == 0:
            continue
        zz, aa, _ = line_score(s)
        zs.append(zz)
        print("   経度 %6.0f〜%6.0f / 緯度 %4.0f〜%4.0f   z = %.2f  (角度 %.0f)"
              % (c[0], c[1], c[2], c[3], zz, aa))
    if zs:
        base = float(np.median(zs))
        print()
        print("対照の中央値 z = %.2f" % base)
        print("対象 / 対照 = %.2f" % (z / max(base, 1e-6)))
        if z / max(base, 1e-6) >= 1.5:
            print(">>> 対象域に人工的な直線がある")
        else:
            print(">>> 対象域に目立つ直線は無い")


if __name__ == "__main__":
    main()
