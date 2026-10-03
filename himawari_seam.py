#!/usr/bin/env python3
"""himawari_seam.py  (試作 v1)
GMGSI の 93.1E の衛星境界（西=Meteosat / 東=ひまわり）で雲が切れる問題を、
ひまわり9号の13番バンド(10.4um)を使って、境界の西側 --width 度ぶんをなだらかにひまわりへ切り替えて直す。

手順:
  1) GMGSI の nc から観測時刻を読み、いちばん近い10分の全球観測を AWS noaa-himawari9 (ISatSS) から88枚取得（キャッシュあり）
  2) 5500x5500 の輝度温度に組み立て、5x5平均で10km格子に
  3) equirect の各画素を静止衛星投影(sweep=y)に変換して読み出す
  4) 境界の東側（GMGSI＝ひまわり由来の所）で、輝度温度→入力PNGの明るさ の対応表を緯度5度ごとに作る
  5) 境界の西 --width 度で 0→1 にひまわりを混ぜる（東側3度で1→0に戻す）。視野の端(天頂角70〜80度)と高緯度は弱める
使い方:
  python3 himawari_seam.py 入力.png 出力.png --nc inputs/latest/*/gmgsi_lw.nc --preview him_preview.png
"""
import argparse
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

BUCKET = "https://noaa-himawari9.s3.amazonaws.com"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PROD = "AHI-L2-FLDK-ISatSS/"


def listing(prefix):
    prefixes, files, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(BUCKET + "/?" + urllib.parse.urlencode(q), timeout=60) as r:
            root = ET.fromstring(r.read())
        prefixes += [e.text for e in root.findall("s:CommonPrefixes/s:Prefix", NS)]
        for c in root.findall("s:Contents", NS):
            files.append((c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)))
        if root.findtext("s:IsTruncated", default="false", namespaces=NS) == "true":
            token = root.findtext("s:NextContinuationToken", namespaces=NS)
        else:
            return prefixes, files


def gmgsi_slot(nc_path):
    import netCDF4
    ds = netCDF4.Dataset(nc_path)
    found = None
    for name, v in ds.variables.items():
        units = getattr(v, "units", "")
        if "since" in str(units):
            try:
                t = netCDF4.num2date(np.ravel(v[:])[0], units)
                print(f"  GMGSI 時刻変数 {name} = {t}")
                found = found or t
            except Exception:
                pass
    for a in ds.ncattrs():
        if "time" in a.lower() or "date" in a.lower():
            print(f"  GMGSI 属性 {a} = {ds.getncattr(a)}")
    ds.close()
    if found is None:
        return None
    m = (found.minute // 10) * 10
    return f"{found.year:04d}{found.month:02d}{found.day:02d}{found.hour:02d}{m:02d}"


def fetch_slot(slot, cache):
    d = os.path.join(cache, slot)
    os.makedirs(d, exist_ok=True)
    _, files = listing(f"{PROD}{slot[:4]}/{slot[4:6]}/{slot[6:8]}/{slot[8:12]}/")
    keys = [k for k, _ in files if "M1C13" in k and k.split("/")[-1].startswith("OR_HFD-")]
    if not keys:
        sys.exit(f"中止: {slot} のC13全球タイルが無い（--slot で別の時刻を指定）")
    paths = []
    for k in keys:
        p = os.path.join(d, k.split("/")[-1])
        if not os.path.exists(p):
            urllib.request.urlretrieve(BUCKET + "/" + k, p)
        paths.append(p)
    print(f"  ひまわり {slot}: タイル {len(paths)} 枚")
    return paths


def assemble(paths):
    import netCDF4
    full = np.full((5500, 5500), np.nan, np.float32)
    geo = None
    for p in paths:
        ds = netCDF4.Dataset(p)
        r0 = int(ds.getncattr("tile_row_offset"))
        c0 = int(ds.getncattr("tile_column_offset"))
        bt = np.ma.filled(ds["Sectorized_CMI"][:].astype(np.float32), np.nan)
        x = np.asarray(ds["x"][:], np.float64)
        y = np.asarray(ds["y"][:], np.float64)
        h, w = bt.shape
        full[r0:r0 + h, c0:c0 + w] = bt
        if geo is None:
            dx, dy = x[1] - x[0], y[1] - y[0]
            geo = (x[0] - c0 * dx, dx, y[0] - r0 * dy, dy)
            proj = ds["fixedgrid_projection"]
            pa = {a: proj.getncattr(a) for a in proj.ncattrs()}
        ds.close()
    full[(full < 150) | (full > 350)] = np.nan
    return full, geo, pa


def block_mean(a, k):
    H, W = a.shape
    a = a[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(1, 3))


def geos_angles(lat, lon, pa):
    a = float(pa["semi_major"]); b = float(pa["semi_minor"])
    h = float(pa["perspective_point_height"]); lon0 = float(pa["longitude_of_projection_origin"])
    sweep = str(pa.get("sweep_angle_axis", "y"))
    rp = b / a; rp2 = rp * rp; rg = 1.0 + h / a
    lam = np.radians(lon - lon0)
    phi = np.arctan(rp2 * np.tan(np.radians(lat)))
    r = rp / np.hypot(rp * np.cos(phi), np.sin(phi))
    vx = r * np.cos(lam) * np.cos(phi)
    vy = r * np.sin(lam) * np.cos(phi)
    vz = r * np.sin(phi)
    vis = ((rg - vx) * vx - vy * vy - vz * vz / rp2) >= 0
    tmp = rg - vx
    if sweep == "x":
        xa = np.arctan(vy / np.hypot(vz, tmp)); ya = np.arctan(vz / tmp)
    else:
        xa = np.arctan(vy / tmp); ya = np.arctan(vz / np.hypot(vy, tmp))
    return xa * 1e6, ya * 1e6, vis


def sat_zenith(lat, lon, lon0=140.7):
    R, Rs = 6371.0, 42164.0
    cg = np.cos(np.radians(lat)) * np.cos(np.radians(lon - lon0))
    d = np.sqrt(R * R + Rs * Rs - 2 * R * Rs * cg)
    return np.degrees(np.arccos(np.clip((Rs * cg - R) / d, -1, 1)))


def main():
    ap = argparse.ArgumentParser(description="ひまわりで93.1Eの衛星境界をなだらかにする（試作）")
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--nc", help="GMGSI の gmgsi_lw.nc（観測時刻を読む）")
    ap.add_argument("--slot", help="ひまわりの時刻 YYYYMMDDHHMM（UTC, 10分刻み）。--nc より優先")
    ap.add_argument("--seam", type=float, default=93.1)
    ap.add_argument("--width", type=float, default=12.0, help="境界の西で混ぜる幅(度)")
    ap.add_argument("--latmax", type=float, default=60.0)
    ap.add_argument("--zmax", type=float, default=80.0, help="ひまわりの天頂角の上限(度)。この10度手前から弱める")
    ap.add_argument("--lattaper", type=float, default=5.0)
    ap.add_argument("--cache", default=os.path.expanduser("~/wallpaper-work/himawari_probe"))
    ap.add_argument("--mincorr", type=float, default=0.85, help="位置合わせの相関がこれ未満なら何も書き出さずに終わる")
    ap.add_argument("--preview")
    a = ap.parse_args()

    slot = a.slot
    if not slot and a.nc:
        slot = gmgsi_slot(a.nc)
    if not slot:
        sys.exit("中止: GMGSI の時刻が読めない。上の属性を見て --slot YYYYMMDDHHMM を指定")
    full, (x0, dx, y0, dy), pa = assemble(fetch_slot(slot, a.cache))
    K = 5
    bt5 = block_mean(full, K)
    x0k, dxk = x0 + (K - 1) / 2 * dx, dx * K
    y0k, dyk = y0 + (K - 1) / 2 * dy, dy * K

    raw = np.asarray(Image.open(a.src))
    g = raw[..., :3].astype(np.float64).mean(axis=2) if raw.ndim == 3 else raw.astype(np.float64)
    H, W = g.shape
    lat1 = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    lon1 = -180.0 + (np.arange(W) + 0.5) / W * 360.0
    cs = np.where((lon1 >= a.seam - a.width - 1) & (lon1 <= a.seam + 32))[0]
    rs = np.where(np.abs(lat1) <= a.latmax + 1)[0]
    LAT, LON = np.meshgrid(lat1[rs], lon1[cs], indexing="ij")

    xa, ya, vis = geos_angles(LAT, LON, pa)
    ci = np.rint((xa - x0k) / dxk).astype(int)
    ri = np.rint((ya - y0k) / dyk).astype(int)
    ok = vis & (ci >= 0) & (ci < bt5.shape[1]) & (ri >= 0) & (ri < bt5.shape[0])
    B = np.full(LAT.shape, np.nan)
    B[ok] = bt5[ri[ok], ci[ok]]
    zen = sat_zenith(LAT, LON)
    valid = np.isfinite(B) & (zen < a.zmax)
    T = g[np.ix_(rs, cs)]

    centers = np.arange(180.0, 322.0, 2.0)
    bands = np.arange(-a.latmax, a.latmax + 0.1, 5.0)
    tables = []
    calm = valid & (LON >= a.seam + 3) & (LON <= a.seam + 30) & (zen < 70)
    for bc in bands:
        m = calm & (np.abs(LAT - bc) <= 5.0)
        if m.sum() < 400:
            tables.append(None); continue
        bt, tg = B[m], T[m]
        idx = np.clip(((bt - 179.0) / 2.0).astype(int), 0, centers.size - 1)
        med = np.full(centers.size, np.nan)
        for i in range(centers.size):
            s = tg[idx == i]
            if s.size >= 8:
                med[i] = np.median(s)
        good = np.isfinite(med)
        if good.sum() < 5:
            tables.append(None); continue
        med = np.interp(centers, centers[good], med[good])
        med = np.minimum.accumulate(med)
        tables.append(med)
    have = [i for i, t in enumerate(tables) if t is not None]
    if not have:
        sys.exit("中止: 対応表が1つも作れない（位置合わせ失敗の疑い）")
    for i in range(len(tables)):
        if tables[i] is None:
            j = min(have, key=lambda k: abs(k - i)); tables[i] = tables[j]
    tables = np.array(tables)

    M = np.full(LAT.shape, np.nan)
    for r in range(LAT.shape[0]):
        p = (LAT[r, 0] - bands[0]) / 5.0
        b0 = int(np.clip(np.floor(p), 0, len(bands) - 2)); w = float(np.clip(p - b0, 0, 1))
        row = B[r]
        f = np.isfinite(row)
        M[r, f] = (1 - w) * np.interp(row[f], centers, tables[b0]) + w * np.interp(row[f], centers, tables[b0 + 1])

    chk = calm & np.isfinite(M)
    mad = float(np.median(np.abs(M[chk] - T[chk])))
    cor = float(np.corrcoef(M[chk], T[chk])[0, 1])
    print(f"  位置合わせの検算（境界の東側）: 相関 {cor:.3f} / 差の中央値 {mad:.1f}")
    if not np.isfinite(cor) or cor < a.mincorr:
        print(f"スキップ: 相関 {cor:.3f} < {a.mincorr}。入力はそのまま（書き出さない）")
        return

    t = np.clip((LON - (a.seam - a.width)) / a.width, 0, 1)
    wl = t * t * (3 - 2 * t)
    e = np.clip((a.seam + 3 - LON) / 3.0, 0, 1)
    wgt = np.where(LON <= a.seam, wl, e)
    wz = np.clip((a.zmax - zen) / 10.0, 0, 1); wz = wz * wz * (3 - 2 * wz)
    wlat = np.clip((a.latmax - np.abs(LAT)) / a.lattaper, 0, 1)
    wgt = wgt * wz * wlat * np.isfinite(M)
    out = g.copy()
    sub = out[np.ix_(rs, cs)]
    sub = np.where(wgt > 0, (1 - wgt) * sub + wgt * np.nan_to_num(M), sub)
    out[np.ix_(rs, cs)] = sub
    west = (LON < a.seam) & (LON > a.seam - a.width)
    print(f"  境界の西{a.width:.0f}度でひまわりが使えた割合: {100 * float((wgt[west] > 0.01).mean()):.1f}%")

    o8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if raw.ndim == 3:
        res = raw.copy()
        for ch in range(min(3, raw.shape[2])):
            res[..., ch] = o8
        Image.fromarray(res).save(a.out)
    else:
        Image.fromarray(o8).save(a.out)
    print(f"書き出し: {a.out}")

    if a.preview:
        deg = W / 360.0
        c = int(round((a.seam + 180.0) / 360.0 * W - 0.5))
        idx = (c + np.arange(-int(15 * deg), int(15 * deg))) % W
        r0, r1 = int(H * 20 / 180), int(H * 160 / 180)
        before = np.clip(np.rint(g[r0:r1][:, idx]), 0, 255).astype(np.uint8)
        after = o8[r0:r1][:, idx]
        hm = np.full((H, W), 90.0)
        hm[np.ix_(rs, cs)] = np.where(np.isfinite(M), M, 90.0)
        him = np.clip(np.rint(hm[r0:r1][:, idx]), 0, 255).astype(np.uint8)
        sep = np.full((before.shape[0], 6), 255, np.uint8)
        pv = Image.fromarray(np.hstack([before, sep, after, sep, him]))
        pv.resize((pv.width * 2, pv.height * 2), Image.NEAREST).save(a.preview)
        print(f"見比べ: {a.preview}（左=前 / 中=後 / 右=ひまわりを明るさ変換したもの。経度±15度・緯度±70度、2倍）")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"スキップ: {e.code}。入力はそのまま")
    except Exception as e:
        print(f"スキップ: {type(e).__name__}: {e}。入力はそのまま")
