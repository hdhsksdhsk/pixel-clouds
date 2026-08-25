"""wrap_ops.py — equirect(経度が周期的)を壊さない画像処理。

経度方向だけラップし、緯度方向は従来通り扱う。
Image.resize も scipy の gaussian_filter も端をラップしないので、
そのまま使うと経度180度(LON_OFFSET=0のとき)に段差が残る。
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter as _gf
from scipy.ndimage import binary_erosion as _be


def wrap_resize(a, out_w, out_h, pad_out=64, resample=Image.LANCZOS):
    """横をラップさせてリサイズする。a は 2D の float 配列。

    パディング幅は「出力側で整数」から決め、元側の幅をそこから逆算する。
    こうしないと丸め誤差で拡張画像の縮尺が元とずれ、切り出した両端1列が
    隣の位置の値を拾って段差になる。
    """
    a = np.ascontiguousarray(a, np.float32)
    src_w = a.shape[1]
    scale = src_w / out_w
    ps = int(round(pad_out * scale))
    ps = max(1, min(ps, src_w // 4))
    pad_out = int(round(ps / scale))          # 元側を基準に取り直す
    ext = np.concatenate([a[:, -ps:], a, a[:, :ps]], axis=1)
    tw = int(round((src_w + 2 * ps) / scale))
    f = Image.fromarray(ext, mode="F")
    r = np.asarray(f.resize((tw, out_h), resample), np.float32)
    off = (tw - out_w) // 2                   # 中央から元の幅を切り出す
    return r[:, off:off + out_w]


def wrap_gauss(a, sigma, **kw):
    """横をラップさせた gaussian_filter。縦は既定(reflect)のまま。"""
    a = np.asarray(a, np.float32)
    s = sigma if np.isscalar(sigma) else max(np.atleast_1d(sigma))
    p = int(np.ceil(4 * float(s))) + 1
    p = min(p, a.shape[1] // 2)
    ext = np.concatenate([a[:, -p:], a, a[:, :p]], axis=1)
    r = _gf(ext, sigma, **kw)
    return r[:, p:p + a.shape[1]]


def wrap_gap(a):
    """東西端の段差の大きさを、通常の隣接列差と比べた比で返す。1.0が理想。"""
    a = np.asarray(a, np.float32)
    col = np.abs(np.diff(a, axis=1)).mean(axis=0)
    gap = np.abs(a[:, 0] - a[:, -1]).mean()
    return float(gap / (np.median(col) + 1e-6))


def wrap_erosion(a, structure=None, iterations=1, **kw):
    """横をラップさせた binary_erosion。
    既定の border_value=0 は経度の東西端を問答無用で無効にするので、
    equirect には使えない。縦は既定のまま(極は本当に端)。"""
    a = np.asarray(a, bool)
    p = max(4, int(iterations) * 4)
    p = min(p, a.shape[1] // 2)
    ext = np.concatenate([a[:, -p:], a, a[:, :p]], axis=1)
    r = _be(ext, structure=structure, iterations=iterations, **kw)
    return r[:, p:p + a.shape[1]]
