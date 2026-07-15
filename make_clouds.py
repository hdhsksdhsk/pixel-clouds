"""GMGSI最新取得 → LW雲抽出 → Google本家とハイブリッド合成 → equirect出力"""
import sys, subprocess, urllib.request, numpy as np
from netCDF4 import Dataset
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter
from datetime import datetime, timezone, timedelta

W, H = 2048, 1024
BUCKET = "https://noaa-gmgsi-pds.s3.amazonaws.com"

def find_latest_lw():
    """S3から最新のGMGSI_LWファイルのKeyを探す"""
    now = datetime.now(timezone.utc)
    for back in range(2, 12):  # 2〜11時間前を新しい順に
        t = now - timedelta(hours=back)
        prefix = f"GMGSI_LW/{t:%Y/%m/%d/%H}/"
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=5"
        try:
            xml = urllib.request.urlopen(url, timeout=30).read().decode()
        except Exception:
            continue
        import re
        keys = re.findall(r"<Key>([^<]+\.nc)</Key>", xml)
        if keys:
            return keys[-1]
    raise RuntimeError("no GMGSI_LW file found")

def main():
    key = find_latest_lw()
    print("using:", key)
    urllib.request.urlretrieve(f"{BUCKET}/{key}", "gmgsi_lw.nc")

    ds = Dataset("gmgsi_lw.nc")
    data = np.ma.filled(ds.variables["data"][0].astype(np.float32), 0)
    valid = data > 0
    lo, hi = np.percentile(data[valid], [3, 97])
    norm = np.clip((data - lo)/(hi-lo+1e-6), 0, 1)
    cloud = np.power(norm, 1.6) * 255
    cloud[~valid] = 0
    gm_raw = np.clip(cloud, 0, 255).astype(np.uint8)

    # ±72.7 → 全球へ配置
    gm = np.asarray(Image.fromarray(gm_raw).resize((W, int(H*72.7/90)), Image.LANCZOS), np.float32)
    pad = (H - gm.shape[0])//2
    gmgsi_full = np.zeros((H, W), np.float32)
    gmgsi_full[pad:pad+gm.shape[0], :] = gm

    # Google本家(極域)と輝度マッチング＋ブレンド
    google_eq = np.asarray(Image.open("google_clouds_eq.png").convert("L"), np.float32)
    if google_eq.shape != (H, W):
        google_eq = np.asarray(Image.fromarray(google_eq.astype(np.uint8)).resize((W,H), Image.LANCZOS), np.float32)
    lat_axis = np.linspace(90, -90, H)
    band = (np.abs(lat_axis) <= 55) & (np.abs(lat_axis) >= 40)
    g_lo, g_hi = np.percentile(google_eq[band], [20, 95])
    m_lo, m_hi = np.percentile(gmgsi_full[band], [20, 95])
    google_adj = np.clip((google_eq - g_lo)/(g_hi-g_lo+1e-6)*(m_hi-m_lo)+m_lo, 0, 255)
    LAT = np.tile(np.abs(lat_axis)[:,None], (1, W))
    w_google = np.clip((LAT - 40) / 15, 0, 1)
    blended = gmgsi_full*(1-w_google) + google_adj*w_google

    _img = Image.fromarray(np.clip(blended,0,255).astype(np.uint8))
    _img = _img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=2))
    _img = ImageEnhance.Contrast(_img).enhance(1.08)
    _img.save("clouds_src.png")
    print("saved clouds_src.png")

if __name__ == "__main__":
    main()
