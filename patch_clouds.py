#!/usr/bin/env python3
# make_clouds.py の「硬い縁」を直すパッチ。
#   使い方 : cd ~/wallpaper-work/clouds_output && python3 patch_clouds.py
#   戻す   : cp make_clouds.py.orig make_clouds.py
import os
import shutil
import sys

P = "make_clouds.py"
FEATHER = 6.0   # 羽根の幅(画素)。1024行=180度なので約1度

if not os.path.exists(P):
    sys.exit("make_clouds.py が見つからない。clouds_output で実行して。")

src = open(P, encoding="utf-8").read()

if "gm_soft" in src:
    sys.exit("すでにパッチ済みに見える(gm_soft がある)。何もしていない。")

edits = [
    # 1) erosion を使うので import を足す
    (
        "def main():",
        "from scipy.ndimage import binary_erosion\n\n\ndef main():",
    ),
    # 2) 欠測マスクを2画素削る。0埋めはやめる(あとで premultiply する)
    (
        '    data = np.where(valid, data, 0.0)\n'
        '    gm_valid_raw = valid.astype(np.float32)\n',

        '    valid = binary_erosion(valid, np.ones((3, 3), bool), iterations=2)\n'
        '    gm_valid_raw = valid.astype(np.float32)\n',
    ),
    # 3) 黒い穴を作らない。縮小用のヘルパを用意する
    (
        '    cloud[~valid] = 0\n'
        '    gm_raw = np.clip(cloud, 0, 255).astype(np.uint8)\n',

        '    NH = int(H * 72.7 / 90)\n'
        '\n'
        '    def _rs(a):\n'
        '        f = Image.fromarray(np.ascontiguousarray(a, np.float32), mode="F")\n'
        '        return np.asarray(f.resize((W, NH), Image.LANCZOS), np.float32)\n',
    ),
    # 4) premultiply して縮小し、被覆率で割り戻す(縁のリンギングが出ない)
    (
        '    gm = np.asarray(Image.fromarray(gm_raw).resize((W, int(H*72.7/90)), Image.LANCZOS), np.float32)\n'
        '    gmv = np.asarray(Image.fromarray((gm_valid_raw*255).astype(np.uint8)).resize((W, int(H*72.7/90)), Image.LANCZOS), np.float32)/255.0\n'
        '    pad = (H - gm.shape[0])//2\n',

        '    gmv = np.clip(_rs(gm_valid_raw), 0.0, 1.0)\n'
        '    gm = np.clip(_rs(cloud * gm_valid_raw) / np.maximum(gmv, 1e-3), 0, 255)\n'
        '    pad = (H - NH) // 2\n',
    ),
    # 5) 硬い gm_ok は正規化用に残し、配合比だけ羽根化した gm_soft を作る
    (
        '    gm_ok[pad:pad+gm.shape[0], :] = np.clip(gmv, 0, 1)\n',

        '    gm_ok[pad:pad+gm.shape[0], :] = np.clip(gmv, 0, 1)\n'
        '    gm_ok = (gm_ok > 0.5).astype(np.float32)\n'
        '    gm_soft = np.clip(gaussian_filter(gm_ok, sigma=%.1f), 0.0, 1.0)\n'
        '    gm_soft = gm_soft * gm_soft * (3 - 2 * gm_soft)\n' % FEATHER,
    ),
    # 6) 穴の中身を低周波で埋める。高周波の差が穴の中で自動的に 0 になる
    (
        '    base = gm_lo*(1 - w_google) + gg_lo*w_google\n',

        '    gmgsi_full = np.where(gm_ok > 0.5, gmgsi_full, gm_lo)\n'
        '    base = gm_lo*(1 - w_google) + gg_lo*w_google\n',
    ),
    # 7) 高周波の配合比を羽根化したものに差し替える
    (
        '* gm_ok\n    blended',
        '* gm_soft\n    blended',
    ),
    # 8) 毎回の生成で段差を数字にする常設ガード
    (
        '    POLE_GAIN = 1.0\n',

        '    _edge = gaussian_filter(gm_ok, 2)\n'
        '    _b = (_edge > 0.02) & (_edge < 0.98)\n'
        '    _lap = np.abs(blended - gaussian_filter(blended, 2))\n'
        '    if _b.sum() > 100:\n'
        '        print("境界の段差比: %.2f  (1.0=段差なし / 2.0超で要注意)"\n'
        '              % (np.percentile(_lap[_b], 99) / (np.percentile(_lap, 99) + 1e-6)))\n'
        '\n'
        '    POLE_GAIN = 1.0\n',
    ),
]

for i, (old, new) in enumerate(edits, 1):
    n = src.count(old)
    if n != 1:
        sys.exit("[%d] %d件一致。中断。書き換えていない。\n--- 探した文字列 ---\n%s" % (i, n, old))
    src = src.replace(old, new)

compile(src, P, "exec")          # 構文が壊れていたらここで落ちる
shutil.copy(P, P + ".orig")
open(P, "w", encoding="utf-8").write(src)
print("patched. 退避 = make_clouds.py.orig")
