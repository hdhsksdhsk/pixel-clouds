"""equirectangular雲画像 → cubemap DDSタイル群 変換パイプライン v2 (bilinear)"""
import os, sys, json
import numpy as np
from PIL import Image
import quicktex.dds
from quicktex.s3tc.bc1 import BC1Encoder

FACE_SIZE = 1024
TILE = 256
GRID = 4
LON_OFFSET = 0.0

def face_dirs(face, u, v):
    o = np.ones_like(u)
    if face == 0:   return  o, -v, -u   # +X (GL standard)
    if face == 1:   return -o, -v,  u   # -X
    if face == 2:   return  u,  o,  v   # +Y
    if face == 3:   return  u, -o, -v   # -Y
    if face == 4:   return  u, -v,  o   # +Z
    if face == 5:   return -u, -v, -o   # -Z

def render_face(eq, face):
    H, W = eq.shape
    idx = (np.arange(FACE_SIZE) + 0.5) / FACE_SIZE * 2 - 1
    u, v = np.meshgrid(idx, idx)
    x, y, z = face_dirs(face, u, v)
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x) + np.radians(LON_OFFSET)
    fx = ((lon + np.pi) / (2 * np.pi)) % 1.0 * W - 0.5
    fy = (np.pi / 2 - lat) / np.pi * (H - 1)
    x0 = np.floor(fx).astype(int)
    wx = fx - x0
    x0 = x0 % W
    x1 = (x0 + 1) % W          # 経度はラップ
    y0 = np.clip(np.floor(fy).astype(int), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wy = fy - np.floor(fy)
    e = eq.astype(np.float32)
    val = (e[y0, x0] * (1 - wx) * (1 - wy) + e[y0, x1] * wx * (1 - wy)
         + e[y1, x0] * (1 - wx) * wy       + e[y1, x1] * wx * wy)
    # 極収束: 高緯度ほど、その緯度帯の平均輝度に寄せて放射状アーティファクトを防ぐ
    abslat = np.abs(lat)
    t = np.clip((np.degrees(abslat) - 78.0) / 12.0, 0, 1)**2  # 78度から極へ0→1
    if t.max() > 0:
        # 各行(=各緯度)の平均をブレンド先にする
        from scipy import ndimage as _nd
        latd = np.degrees(abslat)
        bins = np.clip(((latd - 40) / 2).astype(int), 0, 24)
        means = np.asarray(_nd.mean(val, bins, index=np.arange(25)), dtype=float)
        centers = 41.0 + 2.0*np.arange(25)          # 各ビンの中心緯度
        ok = np.isfinite(means)
        prof = np.interp(latd, centers[ok], means[ok])   # 段差を消す連続補間
        val = val * (1 - t) + prof * t
    return np.clip(val + 0.5, 0, 255).astype(np.uint8)

def patch_header(path):
    data = bytearray(open(path, "rb").read())
    data[0x18:0x20] = b"\x00" * 8
    open(path, "wb").write(data)

def main(src_path, out_dir, date_dir):
    os.makedirs(f"{out_dir}/{date_dir}", exist_ok=True)
    eq = np.asarray(Image.open(src_path).convert("L"))
    print("source:", eq.shape)
    for face in range(6):
        img = render_face(eq, face)
        for col in range(GRID):
            for row in range(GRID):
                tile = img[row*TILE:(row+1)*TILE, col*TILE:(col+1)*TILE]
                rgba = Image.fromarray(tile).convert("RGBA")
                out = f"{out_dir}/{date_dir}/{face}_2_{col}_{row}.dds"
                quicktex.dds.encode(rgba, BC1Encoder(), "DXT1", mip_count=1).save(out)
                patch_header(out)
        print(f"face {face} done")
    with open(f"{out_dir}/root.json", "w") as f:
        json.dump({"baseUrl": f"https://hdhsksdhsk.github.io/pixel-clouds/{date_dir}/"}, f)
    print("all done")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
