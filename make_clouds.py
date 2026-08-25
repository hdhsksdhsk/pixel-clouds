"""GMGSI最新取得 → LW雲抽出 → Google本家とハイブリッド合成 → equirect出力"""
import os, sys, subprocess, urllib.request, numpy as np
from netCDF4 import Dataset
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter
# --- wrap patch ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/wallpaper-work'))
from wrap_ops import wrap_resize as _wrap_resize, wrap_gauss as _wrap_gauss, wrap_gap as _wrap_gap, wrap_erosion as _wrap_erosion
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

from scipy.ndimage import binary_erosion


def main():
    if "--no-download" in sys.argv and os.path.exists("gmgsi_lw.nc"):
        print("using: 既存の gmgsi_lw.nc (--no-download)")   # # --- wrap patch ---
    else:
        key = find_latest_lw()
        print("using:", key)
        urllib.request.urlretrieve(f"{BUCKET}/{key}", "gmgsi_lw.nc")

    ds = Dataset("gmgsi_lw.nc")
    data = np.ma.filled(ds.variables["data"][0].astype(np.float32), 0)
    dqf = np.ma.filled(ds.variables["dqf"][0].astype(np.int16), 1)
    valid = (data > 0) & (dqf == 0)
    print("dqf無効: %.2f%%" % ((~valid).mean()*100))
    valid = _wrap_erosion(valid, np.ones((3, 3), bool), iterations=2)   # --- wrap patch ---
    gm_valid_raw = valid.astype(np.float32)
    lo, hi = np.percentile(data[valid], [3, 97])
    norm = np.clip((data - lo)/(hi-lo+1e-6), 0, 1)
    cloud = np.power(norm, 1.6) * 255
    NH = int(H * 72.7 / 90)

    def _rs(a):
        return _wrap_resize(a, W, NH)   # # --- wrap patch ---

    # ±72.7 → 全球へ配置
    gmv = np.clip(_rs(gm_valid_raw), 0.0, 1.0)
    gm = np.clip(_rs(cloud * gm_valid_raw) / np.maximum(gmv, 1e-3), 0, 255)
    pad = (H - NH) // 2
    gmgsi_full = np.zeros((H, W), np.float32)
    gmgsi_full[pad:pad+gm.shape[0], :] = gm
    gm_ok = np.zeros((H, W), np.float32)
    gm_ok[pad:pad+gm.shape[0], :] = np.clip(gmv, 0, 1)
    gm_ok = (gm_ok > 0.5).astype(np.float32)
    gm_soft = np.clip(_wrap_gauss(gm_ok, sigma=6.0), 0.0, 1.0)
    gm_soft = gm_soft * gm_soft * (3 - 2 * gm_soft)

    # Google本家(極域)と輝度マッチング＋ブレンド
    google_eq = np.asarray(Image.open("google_clouds_eq.png").convert("L"), np.float32)
    if google_eq.shape != (H, W):
        google_eq = np.asarray(Image.fromarray(google_eq.astype(np.uint8)).resize((W,H), Image.LANCZOS), np.float32)
    lat_axis = np.linspace(90, -90, H)
    band = (np.abs(lat_axis) <= 55) & (np.abs(lat_axis) >= 40)
    g_lo, g_hi = np.percentile(google_eq[band], [20, 95])
    m_lo, m_hi = np.percentile(gmgsi_full[band], [20, 95])
    google_adj = np.clip((google_eq - g_lo)/(g_hi-g_lo+1e-6)*(m_hi-m_lo)+m_lo, 0, 255)

    # 継ぎ目(60-71度)の輝度を半球ごとにGMGSIへ合わせる。暗くする方向のみ適用。
    seam = (np.abs(lat_axis) >= 60) & (np.abs(lat_axis) <= 71)
    for hemi in (lat_axis > 0, lat_axis < 0):
        s = seam & hemi
        m0, s0 = google_adj[s].mean(), google_adj[s].std()
        m1, s1 = gmgsi_full[s].mean(), gmgsi_full[s].std()
        if m1 >= m0:
            continue
        g = min(s1/(s0+1e-6), 1.3)
        r = np.where(hemi)[0]
        google_adj[r] = (google_adj[r] - m0)*g + m1
    K = 24.0   # 0付近を滑らかに落としてハードクリップを避ける
    google_adj = np.where(google_adj >= K, google_adj,
                          K*np.exp(np.clip((google_adj - K)/K, -30, 0)))
    google_adj = np.clip(google_adj, 0, 255)

    LAT = np.tile(np.abs(lat_axis)[:,None], (1, W))
    t = np.clip((LAT - 62) / 10, 0, 1)
    w_google = t*t*(3 - 2*t)
    # ±72.7度より外のゼロ詰めが滲まないよう、有効マスクで正規化する
    vmask = gm_ok.copy()
    gm_lo = _wrap_gauss(gmgsi_full, sigma=12) / np.maximum(_wrap_gauss(vmask, sigma=12), 0.05)
    gg_lo = _wrap_gauss(google_adj, sigma=12)
    gmgsi_full = np.where(gm_ok > 0.5, gmgsi_full, gm_lo)
    base = gm_lo*(1 - w_google) + gg_lo*w_google
    detail = np.clip((71 - LAT) / 5, 0, 1) * gm_soft
    blended = base + (gmgsi_full - gm_lo)*detail + (google_adj - gg_lo)*(1 - detail)

    _edge = _wrap_gauss(gm_ok, 2)
    _b = (_edge > 0.02) & (_edge < 0.98)
    _lap = np.abs(blended - _wrap_gauss(blended, 2))
    if _b.sum() > 100:
        print("境界の段差比: %.2f  (1.0=段差なし / 2.0超で要注意)"
              % (np.percentile(_lap[_b], 99) / (np.percentile(_lap, 99) + 1e-6)))

    POLE_GAIN = 1.0
    for _hemi in (lat_axis > 0, lat_axis < 0):
        _ref = blended[(np.abs(lat_axis) >= 66) & (np.abs(lat_axis) < 72) & _hemi]
        _m0, _s0 = float(_ref.mean()), float(_ref.std())
        for _y in np.where((np.abs(lat_axis) >= 72) & _hemi)[0]:
            _w = min((abs(lat_axis[_y]) - 72.0) / 6.0, 1.0)
            _r = blended[_y]
            _m1, _s1 = float(_r.mean()), float(_r.std())
            _g = (1 - _w) + min(_s0 / (_s1 + 1e-6), POLE_GAIN) * _w
            blended[_y] = (_r - _m1) * _g + (_m1 * (1 - _w) + _m0 * _w)

    _img = Image.fromarray(np.clip(blended,0,255).astype(np.uint8))
    _img = ImageEnhance.Contrast(_img).enhance(1.0)
    _img.save("clouds_src.png")
    print("saved clouds_src.png")
    print("東西端の段差比: %.2f  (1.0=段差なし / 1.6超でNG)"
          % _wrap_gap(np.asarray(_img, np.float32)))   # # --- wrap patch ---

if __name__ == "__main__":
    main()
