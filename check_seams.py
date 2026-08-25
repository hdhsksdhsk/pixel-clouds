#!/usr/bin/env python3
"""check_seams.py — 配信前の継ぎ目検査。NGなら終了コード1。

  python3 check_seams.py --eq clouds_output/clouds_src.png
  python3 check_seams.py --tiles clouds_output/20260824_1000
  python3 check_seams.py --eq clouds_output/clouds_src.png --tiles clouds_output/20260824_1000
  python3 check_seams.py --faces devcache/cache        (端末が作った px.jpg 等)

検査するもの:
  equirect : 東西端(経度のラップ)の段差比 / 突出した列・行
  面       : 面ごとの突出した列・行(直線状のアーティファクト)

★2026-08-25 に判明した経緯（同じ道を二度通らないために）
  ・端末の雲は 96枚の .dds を JNI(ImageMagick) が面ごとに貼り合わせ、
    1024^2 の JPEG にしてから読む。.jpg はサーバーには無い。
  ・「日本の東の細い線」の正体は経度180度＝equirect の東西端の段差。
    赤道でもタイル境界でも面境界でもなかった。
  ・原因は make_clouds.py の LANCZOS リサイズと gaussian_filter が
    経度をラップしないこと。元の GMGSI データは繋がっている(比1.15)。
  ・無罪が確定しているもの: ステッチャ / BC1圧縮 / JPEG再エンコード /
    シェーダーのパララックス / 面境界 / タイル境界。もう疑わない。
"""
import argparse
import os
import struct
import sys

import numpy as np
from PIL import Image

NG_WRAP = 1.8      # 東西端の段差比。これを超えたらNG
                   # 既知: 平常時 1.46-1.55（GMGSI側由来。google素材の端を直しても0.09しか動かない）
                   # 実害があった日付変更線バグは 2.44 だった。1.8 でも十分捕まる
WARN_WRAP = 1.3
NG_Z = 7.0         # 列/行の突出。これを超えたらNG
                   # 既知: 列1553 (経度約93E) が z=5.6-6.2。GMGSIモザイク内部の衛星境界で元データ由来
                   # 実害があったバグは列0が z=10.8 だった
WARN_Z = 4.5

FACE_NAMES = ["px", "nx", "py", "ny", "pz", "nz"]


def line_scan(g):
    """行と列それぞれの、隣接差の突出度(z)の最大値と位置を返す。"""
    out = {}
    for axis, name in ((0, "行"), (1, "列")):
        d = np.abs(np.diff(g, axis=axis)).mean(axis=1 - axis)
        z = (d - d.mean()) / (d.std() + 1e-9)
        i = int(np.argmax(z))
        out[name] = (i, float(z[i]))
    return out


def wrap_ratio(g):
    col = np.abs(np.diff(g, axis=1)).mean(axis=0)
    gap = np.abs(g[:, 0] - g[:, -1]).mean()
    return float(gap / (np.median(col) + 1e-6))


def gray(path):
    return np.asarray(Image.open(path).convert("L"), np.float32)


def bc1_decode(b):
    """DDS(BC1) 256x256 を RGB に展開する。ヘッダ128バイト固定。"""
    n = 256
    out = np.zeros((n, n, 3), np.uint8)
    o = 128
    for by in range(n // 4):
        for bx in range(n // 4):
            c0, c1 = struct.unpack_from("<HH", b, o)
            bits = struct.unpack_from("<I", b, o + 4)[0]
            o += 8

            def col(c):
                return np.array([(c >> 11 & 31) * 255 // 31,
                                 (c >> 5 & 63) * 255 // 63,
                                 (c & 31) * 255 // 31], np.float32)

            a, bb = col(c0), col(c1)
            if c0 > c1:
                p = [a, bb, (2 * a + bb) / 3, (a + 2 * bb) / 3]
            else:
                p = [a, bb, (a + bb) / 2, np.zeros(3)]
            for y in range(4):
                for x in range(4):
                    out[by * 4 + y, bx * 4 + x] = p[(bits >> (2 * (y * 4 + x))) & 3]
    return out


def face_from_tiles(d, f):
    face = np.zeros((1024, 1024, 3), np.uint8)
    for r in range(4):
        for c in range(4):
            p = os.path.join(d, f"{f}_2_{c}_{r}.dds")
            if not os.path.exists(p):
                return None
            face[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = bc1_decode(open(p, "rb").read())
    return face.mean(axis=2)


def verdict(v, warn, ng):
    return "NG" if v > ng else ("警告" if v > warn else "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", help="equirect の画像 (clouds_src.png)")
    ap.add_argument("--tiles", help="96枚の .dds が入っているディレクトリ")
    ap.add_argument("--faces", help="面の画像が入っているディレクトリ (px.jpg 等)")
    a = ap.parse_args()

    if not (a.eq or a.tiles or a.faces):
        ap.print_help()
        sys.exit(2)

    bad = False

    if a.eq:
        g = gray(a.eq)
        w = wrap_ratio(g)
        v = verdict(w, WARN_WRAP, NG_WRAP)
        bad |= v == "NG"
        print(f"[equirect] {a.eq}  {g.shape}")
        print(f"    東西端の段差比 {w:.2f}   {v}   (1.0=段差なし)")
        s = line_scan(g)
        for k, (i, z) in s.items():
            vv = verdict(z, WARN_Z, NG_Z)
            bad |= vv == "NG"
            print(f"    突出した{k} {i}  z={z:.1f}   {vv}")

    if a.tiles:
        print(f"[タイル] {a.tiles}")
        for fi, fn in enumerate(FACE_NAMES):
            g = face_from_tiles(a.tiles, fi)
            if g is None:
                print(f"    face{fi} ({fn}): タイルが揃っていない   NG")
                bad = True
                continue
            s = line_scan(g)
            msgs = []
            for k, (i, z) in s.items():
                vv = verdict(z, WARN_Z, NG_Z)
                bad |= vv == "NG"
                msgs.append(f"{k}{i} z={z:.1f} {vv}")
            print(f"    face{fi} ({fn}): " + "  ".join(msgs))

    if a.faces:
        print(f"[面の画像] {a.faces}")
        for fn in FACE_NAMES:
            for ext in (".jpg", ".png"):
                p = os.path.join(a.faces, fn + ext)
                if os.path.exists(p):
                    break
            else:
                print(f"    {fn}: 無い")
                continue
            g = gray(p)
            s = line_scan(g)
            msgs = []
            for k, (i, z) in s.items():
                vv = verdict(z, WARN_Z, NG_Z)
                bad |= vv == "NG"
                msgs.append(f"{k}{i} z={z:.1f} {vv}")
            print(f"    {fn}: " + "  ".join(msgs))

    print("---- 判定 ----")
    if bad:
        print("NG  配信しないこと")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
