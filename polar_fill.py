"""polar_fill.py — 北極（北緯60度より北）を、北極を通る衛星(NASA GIBS の MODIS/VIIRS 赤外)で埋める

make_clouds.py から呼ばれる。失敗したら例外を投げるだけ（呼び出し側が今まで通りの絵を使う）。
  ・各レイヤーは「今日の分」を優先し、まだ撮れていない所は「きのうの分」で埋める
  ・全レイヤーの中央値をとって、撮った帯の継ぎ目を目立たなくする
  ・緯度ごとの背景の冷たさを取り除き（明るさ補正）、北緯60〜66度の帯で今の絵と明るさの分布を揃えてなだらかに切り替える
"""
import io, math, os, re, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter1d

WMS = "https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi"
WMTS_CAPS = "https://gibs.earthdata.nasa.gov/wmts/epsg3413/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities"
EXTENT = 4194304.0
LON0 = -45.0
WANT = re.compile(r"Brightness_Temp_Band(31|I5)_(Day|Night)$")
NS = {"wmts": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1",
      "xlink": "http://www.w3.org/1999/xlink"}
UA = {"User-Agent": "pixel-clouds/polar_fill (personal wallpaper project)"}
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
MIN_LAYERS = 4


def _get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _layers():
    root = ET.fromstring(_get(WMTS_CAPS))
    out = {}
    for lay in root.iter("{%s}Layer" % NS["wmts"]):
        ident = lay.find("ows:Identifier", NS)
        if ident is None or not WANT.search(ident.text or ""):
            continue
        for md in lay.findall("ows:Metadata", NS):
            role = md.get("{%s}role" % NS["xlink"], "")
            if "colormap" in role and role.endswith("1.3"):
                out[ident.text] = md.get("{%s}href" % NS["xlink"])
    return out


def _colormap(xml_bytes):
    lut = {}
    for e in ET.fromstring(xml_bytes).iter("ColorMapEntry"):
        if e.get("transparent") == "true" or e.get("nodata") == "true":
            continue
        nums = [float(x) for x in NUM.findall(e.get("value") or "")]
        if e.get("rgb") and nums:
            lut[tuple(int(c) for c in e.get("rgb").split(","))] = sum(nums) / len(nums)
    return lut


def _kelvin(img, lut):
    a = np.asarray(img.convert("RGBA"))
    key = (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2].astype(np.int32)
    keys = np.array([(r << 16) | (g << 8) | b for (r, g, b) in lut], np.int32)
    vals = np.array(list(lut.values()), np.float32)
    o = np.argsort(keys)
    keys, vals = keys[o], vals[o]
    i = np.clip(np.searchsorted(keys, key), 0, len(keys) - 1)
    hit = (keys[i] == key) & (a[..., 3] > 0)
    t = np.where(hit, vals[i], np.nan).astype(np.float32)
    f = t[np.isfinite(t)]
    if f.size and np.median(f) < 150:
        t = t + 273.15
    return t


def _fetch(name, date, size, lut):
    url = (f"{WMS}?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={name}&STYLES="
           f"&CRS=EPSG:3413&BBOX={-EXTENT},{-EXTENT},{EXTENT},{EXTENT}"
           f"&WIDTH={size}&HEIGHT={size}&FORMAT=image/png&TRANSPARENT=TRUE&TIME={date}")
    return _kelvin(Image.open(io.BytesIO(_get(url, timeout=180))), lut)


def build_composite(size=1024, now=None, log=print):
    now = now or datetime.now(timezone.utc)
    today, yday = now.date().isoformat(), (now.date() - timedelta(days=1)).isoformat()
    layers = _layers()
    stack, n_today = [], 0
    for name, cm in sorted(layers.items()):
        try:
            lut = _colormap(_get(cm))
        except Exception as e:
            log(f"polar:   skip {name}: 色の対応表 {e}")
            continue
        t_y = t_t = None
        try:
            t_y = _fetch(name, yday, size, lut)
        except Exception as e:
            log(f"polar:   {name} きのう分なし: {e}")
        try:
            t_t = _fetch(name, today, size, lut)
        except Exception:
            pass
        if t_t is not None and np.isfinite(t_t).any():
            n_today += 1
            t = t_t if t_y is None else np.where(np.isfinite(t_t), t_t, t_y)
        elif t_y is not None:
            t = t_y
        else:
            continue
        stack.append(t)
    if len(stack) < MIN_LAYERS:
        raise RuntimeError(f"取れたレイヤーが少ない ({len(stack)} < {MIN_LAYERS})")
    with np.errstate(all="ignore"):
        comp = np.nanmedian(np.stack(stack), axis=0).astype(np.float32)
    log(f"polar: レイヤー {len(stack)} 枚（今日の分あり {n_today} 枚）  {yday}〜{today}")
    return comp


def _sample(comp, lat, lon):
    n = comp.shape[0]
    R = 6378137.0
    phic = math.radians(70.0)
    k = R * math.cos(phic) / math.tan(math.pi / 4 - phic / 2)
    rho = k * np.tan(np.pi / 4 - np.radians(lat) / 2)
    d = np.radians(lon - LON0)
    x, y = rho * np.sin(d), -rho * np.cos(d)
    col = (x + EXTENT) / (2 * EXTENT) * n - 0.5
    row = (EXTENT - y) / (2 * EXTENT) * n - 0.5
    ok = np.isfinite(comp)
    v = map_coordinates(np.where(ok, comp, 0.0), [row, col], order=1, mode="constant", cval=0.0)
    w = map_coordinates(ok.astype(np.float32), [row, col], order=1, mode="constant", cval=0.0)
    return np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan).astype(np.float32)


def blend(src, lat_axis, comp, lat0=60.0, lat1=66.0, log=print):
    H, W = src.shape
    lon_axis = (np.arange(W) + 0.5) * 360.0 / W - 180.0
    LON, LAT = np.meshgrid(lon_axis, lat_axis)
    T = np.full((H, W), np.nan, np.float32)
    north = LAT >= lat0 - 2
    T[north] = _sample(comp, LAT[north], LON[north])

    cover = np.isfinite(T)[LAT >= lat1].mean() * 100
    if cover < 99.0:
        raise RuntimeError(f"北緯{lat1:g}度より北の埋まり方が足りない ({cover:.1f}%)")

    rowmed = np.array([np.nanmedian(T[r]) if np.isfinite(T[r]).any() else np.nan for r in range(H)])
    good = np.isfinite(rowmed)
    sm = rowmed.copy()
    sm[good] = gaussian_filter1d(rowmed[good], 3.0)
    ref = float(np.mean(sm[(lat_axis >= lat0) & (lat_axis <= lat1) & good]))
    Td = T - (np.where(good, sm, ref) - ref)[:, None]

    band = (LAT >= lat0) & (LAT <= lat1) & np.isfinite(Td)
    q = np.linspace(0.5, 99.5, 199)
    V = np.interp(Td, np.percentile(Td[band], q), np.percentile(src[band], 100 - q)).astype(np.float32)
    t = np.clip((LAT - lat0) / (lat1 - lat0), 0, 1)
    w = t * t * (3 - 2 * t) * np.isfinite(Td)
    out = np.clip(np.where(w > 0, src * (1 - w) + np.nan_to_num(V) * w, src), 0, 255)

    cap = LAT >= 72
    m_before, m_after = float(src[cap].mean()), float(out[cap].mean())
    if not (40.0 <= m_after <= 200.0):
        raise RuntimeError(f"北極の明るさが不自然 ({m_after:.1f})")
    log(f"polar: 埋まり {cover:.1f}%  北緯72度以北の平均 {m_before:.1f} → {m_after:.1f}")
    return out.astype(np.float32)


def apply(src, lat_axis, cache="polar_K.npy", reuse=False, log=print):
    """src: make_clouds の最終画像(H×W, 0-255)。北極を差し替えた新しい配列を返す。"""
    if reuse and os.path.exists(cache):
        comp = np.load(cache)
        log(f"polar: 保存済みの {cache} を再利用")
    else:
        comp = build_composite(log=log)
        np.save(cache, comp)
    return blend(np.asarray(src, np.float32), np.asarray(lat_axis, np.float64), comp, log=log)


def _pole_view(img, n=1024, outer=45.0):
    H, W = img.shape
    c = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(c, c)
    r = np.hypot(u, v)
    lat = 90.0 - r * (90.0 - outer)
    lon = np.degrees(np.arctan2(u, v))
    row = (90.0 - lat) / 180.0 * (H - 1)
    col = ((lon + 180.0) / 360.0 * W - 0.5) % W
    ext = np.concatenate([img, img[:, :1]], axis=1)
    g = map_coordinates(ext, [row, col], order=1, mode="nearest")
    g[r > 1] = 0
    return g


def save_compare(before, after, path="polar_compare.png"):
    """北極を真上から見た 左=差し替え前 / 右=差し替え後 を保存（確認用）"""
    b, a = np.asarray(before, np.float32), np.asarray(after, np.float32)
    pv = np.concatenate([_pole_view(b), np.full((1024, 8), 255.0), _pole_view(a)], axis=1)
    Image.fromarray(np.clip(pv, 0, 255).astype(np.uint8)).save(path)
