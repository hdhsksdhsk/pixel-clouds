#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
google_clouds_eq.png の東西端の壊れた列を修復する。

equirect の列0 と列W-1 は経度上で隣接しているのに、素材自体が端で段差を
持っている（隣接差の中央値 0.00 に対して端は 7〜11）。ここを内側の健全な列
から線形に繋ぎ直す。

  --apply   修復して上書き（退避は ~/wallpaper-work/bak_googleeq/）
  --revert  退避から戻す
  引数なし  状態表示のみ（端の隣接差を出す）
  --width N 片側で置換する列数（既定 2）
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

TARGET = "google_clouds_eq.png"
BAKDIR = os.path.expanduser("~/wallpaper-work/bak_googleeq")
BAK = os.path.join(BAKDIR, TARGET)
MARK = os.path.join(BAKDIR, ".applied")


def load(path):
    im = Image.open(path)
    return im, np.asarray(im, dtype=np.float64)


def report(a, tag):
    """端の隣接差と、内部の中央値を出す。"""
    if a.ndim == 3:
        g = a.mean(axis=2)
    else:
        g = a
    d = np.abs(np.diff(g, axis=1))
    med = float(np.median(d))
    wrap = float(np.abs(g[:, -1] - g[:, 0]).mean())
    e = lambda i, j: float(np.abs(g[:, i] - g[:, j]).mean())
    print("  [%s] 内部med %.2f | wrap(-1,0) %.2f  0-1 %.2f  1-2 %.2f  2-3 %.2f  -1--2 %.2f  -2--3 %.2f"
          % (tag, med, wrap, e(0, 1), e(1, 2), e(2, 3), e(-1, -2), e(-2, -3)))
    return wrap, med


def fix(a, width):
    """端 width 列ずつ（計 2*width 列）を、その外側の健全な列で線形に繋ぐ。

    経度の並びは ... W-3, W-2, W-1 | 0, 1, 2 ... と環になっている。
    置換するのは [-width:] と [:width]。両端の錨は a[:, -width-1] と a[:, width]。
    """
    b = a.copy()
    left_anchor = a[:, -width - 1]    # 置換帯の左の健全な列
    right_anchor = a[:, width]        # 置換帯の右の健全な列
    n = 2 * width + 1                 # 錨から錨までの区間数
    idx = list(range(-width, 0)) + list(range(0, width))
    for k, col in enumerate(idx):
        t = (k + 1) / n
        b[:, col] = left_anchor * (1.0 - t) + right_anchor * t
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--width", type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit("見つからない: %s（clouds_output/ で実行すること）" % TARGET)

    applied = os.path.exists(MARK)

    if args.revert:
        if not os.path.exists(BAK):
            sys.exit("退避が無い: %s" % BAK)
        shutil.copy2(BAK, TARGET)
        if applied:
            os.remove(MARK)
        print("revert 完了")
        _, a = load(TARGET)
        report(a, "戻した後")
        return

    if not args.apply:
        print("状態: %s" % ("適用済み" if applied else "未適用"))
        print("退避: %s (%s)" % (BAK, "あり" if os.path.exists(BAK) else "なし"))
        _, a = load(TARGET)
        report(a, "現在")
        return

    if applied:
        sys.exit("すでに適用済み。やり直すなら先に --revert")

    os.makedirs(BAKDIR, exist_ok=True)
    shutil.copy2(TARGET, BAK)

    im, a = load(TARGET)
    print("%s  mode=%s  shape=%s  width=%d" % (TARGET, im.mode, a.shape, args.width))
    report(a, "修復前")

    b = fix(a, args.width)
    report(b, "修復後")

    out = np.clip(np.rint(b), 0, 255).astype(np.uint8)
    Image.fromarray(out, mode=im.mode).save(TARGET)
    open(MARK, "w").write("width=%d\n" % args.width)
    print("apply 完了（退避 %s）" % BAK)


if __name__ == "__main__":
    main()
