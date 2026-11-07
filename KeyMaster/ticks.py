# Reading the certified scale in one image.
#
#   1. Normalise the image by a smooth background and find the chrome polarity.
#   2. Find the scale baseline (a long straight chrome line) and its angle; it defines the
#      scale axis e and the anchor point (baseline at the image-centre column).
#   3. Find the rows of the tick band, average them into a 1D profile along the axis and
#      locate every tick to sub-pixel precision.
#   4. Measure each tick's length and number the ticks: the long-tick hierarchy of the scale
#      profile plus the approximate scale reading picks the absolute index k.
#   5. Fit certified position X(k) against the along-axis pixel coordinate s:
#      X = A + B s + C s^2 (order set in config). B is the local um/px.
#
# Coordinates: image x to the right, y down. For a point (x, y),
#   s = (x - xc) cos(theta) + (y - y0) sin(theta)     along the scale axis, px
#   t = -(x - xc) sin(theta) + (y - y0) cos(theta)    perpendicular, px (down in the image)
# with (xc, y0) the anchor. Ticks are taken as perpendicular to the baseline.

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy import signal

# image edges give all-NaN columns in the rectified samples; they are handled explicitly
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='All-NaN slice encountered')


# ---------------------------------------------------------------- normalisation

def find_polarity(img):
    """'dark' when the chrome is darker than the background (transmitted light)."""
    sub = img[::4, ::4].astype(np.float64)
    med = np.median(sub)
    mad = np.median(np.abs(sub - med)) * 1.4826 + 1e-9
    z = (sub - med) / mad
    return 'dark' if np.count_nonzero(z < -8) >= np.count_nonzero(z > 8) else 'bright'


def smooth_background(img, polarity, f=8):
    """Background level from block percentiles that ignore the thin chrome lines."""
    H, W = img.shape
    h, w = H // f, W // f
    blocks = img[:h * f, :w * f].reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h, w, f * f)
    q = 75 if polarity == 'dark' else 25
    small = np.percentile(blocks, q, axis=-1)
    small = ndi.median_filter(small, size=7, mode='nearest')
    small = ndi.gaussian_filter(small, 1.0, mode='nearest')
    yy = (np.arange(H) + 0.5) / f - 0.5
    xx = (np.arange(W) + 0.5) / f - 0.5
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    return ndi.map_coordinates(small, [Y, X], order=1, mode='nearest')


def normalise(img, polarity=None, flat=None):
    """Returns (norm, ink, polarity): norm = img/background - 1, ink > 0 on chrome."""
    img = img.astype(np.float64)
    if flat is not None:
        flat = ndi.gaussian_filter(flat.astype(np.float64), 2.0)
        img = img / np.maximum(flat, 1e-9) * np.mean(flat)
    if polarity in (None, '', 'auto'):
        polarity = find_polarity(img)
    bg = smooth_background(img, polarity)
    norm = img / np.maximum(bg, 1e-9) - 1.0
    norm -= np.median(norm[::4, ::4])
    ink = -norm if polarity == 'dark' else norm
    return norm, ink, polarity


# ---------------------------------------------------------------- sampling helpers

def sample_rect(img, x0, y0, theta, s, t, order=1):
    """img sampled at (x0, y0) + s e + t e_perp; returns [len(t), len(s)], NaN outside."""
    c, sn = np.cos(theta), np.sin(theta)
    S, T = np.meshgrid(np.asarray(s, float), np.asarray(t, float))
    X = x0 + S * c - T * sn
    Y = y0 + S * sn + T * c
    H, W = img.shape
    ylo = int(max(0, np.floor(np.nanmin(Y)) - 4))
    yhi = int(min(H, np.ceil(np.nanmax(Y)) + 5))
    if yhi <= ylo:
        return np.full(S.shape, np.nan)
    crop = img[ylo:yhi]
    out = ndi.map_coordinates(crop, [Y - ylo, X], order=order, mode='nearest')
    outside = (X < 0) | (X > W - 1) | (Y < 0) | (Y > H - 1)
    out[outside] = np.nan
    return out


# ---------------------------------------------------------------- baseline

def _line_candidates(ink, max_angle_deg, n_strips=24):
    H, W = ink.shape
    edges = np.linspace(0, W, n_strips + 1).astype(int)
    xc = (W - 1) / 2
    pts = []
    for j in range(n_strips):
        prof = ink[:, edges[j]:edges[j + 1]].mean(axis=1)
        hp = prof - ndi.median_filter(prof, 41, mode='nearest')
        peaks, _ = signal.find_peaks(hp, height=0.25, distance=6)
        for p in peaks:
            pts.append((j, (edges[j] + edges[j + 1] - 1) / 2 - xc, float(p)))
    if not pts:
        return []
    pts = np.array(pts)
    best = []
    slopes = np.tan(np.radians(np.linspace(-max_angle_deg, max_angle_deg,
                                           int(2 * max_angle_deg / 0.02) + 1)))
    for m in slopes:
        icpt = pts[:, 2] - m * pts[:, 1]
        bins = np.round(icpt / 3).astype(int)
        for b in np.unique(bins):
            sel = np.abs(bins - b) <= 1
            votes = len(np.unique(pts[sel, 0]))
            best.append((votes, m, float(np.median(icpt[sel]))))
    best.sort(key=lambda v: -v[0])
    out = []
    for votes, m, c in best:
        if votes < max(4, 0.4 * n_strips):
            break
        if all(abs(c - c2) > 20 for _, _, c2 in out):
            out.append((votes, m, c))
        if len(out) >= 6:
            break
    return out


def _tick_power(ink, xc, y0, theta, s, t_rows, f_tick):
    R = sample_rect(ink, xc, y0, theta, s, t_rows, order=1)
    R = np.nan_to_num(R - np.nanmean(R, axis=1, keepdims=True))
    F = np.abs(np.fft.rfft(R, axis=1))
    n = R.shape[1]
    k0 = f_tick * n
    lo, hi = int(np.floor(k0 * 0.96)), int(np.ceil(k0 * 1.04)) + 1
    if hi >= F.shape[1] or lo < 1:
        return np.zeros(len(t_rows))
    return F[:, lo:hi].max(axis=1) / n


def find_baseline(ink, mpp_nom, division_um, max_angle_deg=3.0):
    H, W = ink.shape
    xc = (W - 1) / 2
    cands = _line_candidates(ink, max_angle_deg)
    if not cands:
        raise RuntimeError("no scale baseline found (no long straight chrome line)")
    f_tick = mpp_nom / division_um
    s = np.arange(-W / 2, W / 2)
    t_rows = np.r_[-40:-5:3, 6:41:3]
    scored = []
    for votes, m, c in cands:
        theta = np.arctan(m)
        pw = _tick_power(ink, xc, c, theta, s, t_rows, f_tick)
        scored.append((np.nanmax(pw), theta, c))
    scored.sort(key=lambda v: -v[0])
    _, theta, y0 = scored[0]
    # refine: centroid of the baseline across the image in strips, then a straight-line fit
    for _ in range(3):
        n_strips = 32
        s_edges = np.linspace(-W / 2 + 20, W / 2 - 20, n_strips + 1)
        t = np.arange(-8, 8.01, 0.25)
        sc, tc = [], []
        for j in range(n_strips):
            ss = np.arange(s_edges[j], s_edges[j + 1])
            R = sample_rect(ink, xc, y0, theta, ss, t, order=1)
            prof = np.nanmean(R, axis=1)
            if np.any(np.isnan(prof)):
                continue
            prof = prof - np.median(np.r_[prof[:8], prof[-8:]])
            core = np.abs(t) <= 4
            wgt = np.clip(prof[core], 0, None)
            if wgt.sum() <= 0:
                continue
            sc.append(ss.mean())
            tc.append(np.sum(wgt * t[core]) / wgt.sum())
        sc, tc = np.array(sc), np.array(tc)
        if len(sc) < 6:
            raise RuntimeError("baseline refinement failed")
        keep = np.ones(len(sc), bool)
        for _ in range(3):
            coef = np.polyfit(sc[keep], tc[keep], 1)
            res = tc - np.polyval(coef, sc)
            mad = np.median(np.abs(res[keep])) * 1.4826 + 0.02
            keep = np.abs(res) < 4 * mad
        # t offset at s: shift the anchor along the normal, rotate by the residual slope
        dtheta = np.arctan(coef[0])
        y0 = y0 + coef[1] / np.cos(theta)
        theta = theta + dtheta
    return theta, y0, float(np.std(res[keep]))


# ---------------------------------------------------------------- ticks

def _centroid(P, c0, hw, half_spacing, iters=6):
    n = len(P)
    c = float(c0)
    for _ in range(iters):
        lo, hi = c - hw, c + hw
        i0, i1 = int(np.floor(lo)), int(np.ceil(hi))
        b0, b1 = int(np.floor(c - half_spacing)), int(np.ceil(c + half_spacing))
        if i0 < 0 or i1 >= n or b0 < 0 or b1 >= n:
            return np.nan
        idx = np.arange(i0, i1 + 1)
        wedge = np.clip(np.minimum(idx - lo + 0.5, hi - idx + 0.5), 0, 1)
        bg = np.percentile(P[b0:b1 + 1], 20)
        v = np.clip(P[idx] - bg, 0, None) * wedge
        if v.sum() <= 0:
            return np.nan
        c_new = np.sum(idx * v) / v.sum()
        if abs(c_new - c) < 1e-5:
            c = c_new
            break
        c = c_new
    return c


def _level_score(lengths, levels):
    """Share of the tick-length variance explained by the predicted levels (eta squared),
    minus 1 when the level means do not increase with level. NaN if there is no contrast."""
    if np.ptp(levels) == 0 or np.ptp(lengths) == 0:
        return np.nan
    # robust per-level centre: median; one odd tick (cross, dirt) must not swing the score
    u = np.unique(levels)
    med = np.array([np.median(lengths[levels == v]) for v in u])
    pred = med[np.searchsorted(u, levels)]
    tot = np.sum((lengths - np.median(lengths)) ** 2)
    within = np.sum(np.minimum((lengths - pred) ** 2, np.percentile((lengths - pred) ** 2, 95)))
    eta = 1.0 - within / max(tot, 1e-12)
    return eta - (0.0 if np.all(np.diff(med) > 0) else 1.0)


@dataclass
class TickResult:
    polarity: str
    theta: float
    xc: float
    y0: float
    baseline_rms_px: float
    band_t: np.ndarray
    s: np.ndarray              # tick centres along the axis, px (all detected)
    length_px: np.ndarray
    k: np.ndarray
    X_um: np.ndarray           # certified positions
    used: np.ndarray           # in the fit
    coef: np.ndarray           # X = polyval(coef, s) (highest power first)
    coef_lin: np.ndarray
    resid_um: np.ndarray
    rms_um: float
    assign_score: float
    assign_margin: float
    sign: int                  # +1 if X increases with s
    flags: list = field(default_factory=list)

    @property
    def mpp(self):
        return abs(np.polyval(np.polyder(self.coef), 0.0))

    def st(self, x, y):
        c, sn = np.cos(self.theta), np.sin(self.theta)
        dx, dy = np.asarray(x) - self.xc, np.asarray(y) - self.y0
        return dx * c + dy * sn, -dx * sn + dy * c

    def X_at(self, s, linear=False):
        return np.polyval(self.coef_lin if linear else self.coef, s)

    def dXds(self, s):
        return np.polyval(np.polyder(self.coef), s)

    @property
    def e(self):
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    @property
    def e_perp(self):
        return np.array([-np.sin(self.theta), np.cos(self.theta)])

    def xy(self, s, t):
        c, sn = np.cos(self.theta), np.sin(self.theta)
        return self.xc + s * c - t * sn, self.y0 + s * sn + t * c


def _fit_scale(s, X, order, sigma_clip=3.0):
    keep = np.isfinite(s) & np.isfinite(X)
    for _ in range(5):
        coef = np.polyfit(s[keep], X[keep], order)
        res = X - np.polyval(coef, s)
        sd = np.median(np.abs(res[keep])) * 1.4826 + 1e-4
        new = keep & (np.abs(res) < sigma_clip * max(sd, 0.01))
        if new.sum() == keep.sum():
            break
        keep = new
    return coef, res, keep


def read_scale(ink, polarity, profile, cert, mpp_nom, scale_mm_approx, increases_to='right',
               fit_order=2, search_mm=0.3, max_angle_deg=3.0):
    H, W = ink.shape
    div = profile.division_um
    flags = []
    theta, y0, base_rms = find_baseline(ink, mpp_nom, div, max_angle_deg)
    xc = (W - 1) / 2
    spacing_nom = div / mpp_nom
    f_tick = 1.0 / spacing_nom

    # tick band: rows (t) with strong periodicity at the tick frequency, off the baseline
    T = int(round(300.0 / mpp_nom))
    t_all = np.arange(-T, T + 1)
    s = np.arange(-W / 2 - 50, W / 2 + 50)
    power = _tick_power(ink, xc, y0, theta, s, t_all, f_tick)
    excl = profile.line_width_um / mpp_nom + 2.5
    off = np.abs(t_all) > excl
    pmax = power[off].max()
    band = off & (power >= 0.6 * pmax)
    band_t = t_all[band]
    if len(band_t) < 3:
        raise RuntimeError("tick band not found")

    # averaged profile along the axis (cubic interpolation)
    Rb = sample_rect(ink, xc, y0, theta, s, band_t, order=3)
    P = np.nanmean(Rb, axis=0)
    valid = np.isfinite(P)
    P = np.where(valid, P, 0.0)
    peaks, _ = signal.find_peaks(P, distance=max(2, int(0.6 * spacing_nom)),
                                 prominence=0.25 * np.nanpercentile(P, 99.5))
    hw = max(2.0, 0.75 * profile.line_width_um / mpp_nom + 1.5)
    hw = min(hw, 0.4 * spacing_nom)
    cents = np.array([_centroid(P, p, hw, 0.5 * spacing_nom) for p in peaks])
    cents = cents[np.isfinite(cents)]
    s_t = s[0] + cents                      # s coordinate (s grid is unit-spaced)
    if len(s_t) < 8:
        raise RuntimeError(f"only {len(s_t)} ticks found")

    # relative indices from the spacing
    spacing = np.median(np.diff(s_t))
    j = np.rint((s_t - s_t[0]) / spacing).astype(int)
    _, first = np.unique(j, return_index=True)
    if len(first) < len(j):
        flags.append("duplicate tick detections removed")
        s_t, j = s_t[first], j[first]
    slope, icpt = np.polyfit(j, s_t, 1)

    # tick lengths along t, contiguous run through the baseline
    R1 = sample_rect(ink, xc, y0, theta, s, t_all, order=1)
    lengths = np.empty(len(s_t))
    lvl = np.nanmedian(Rb, axis=0)
    i0 = np.argmin(np.abs(t_all))
    for n, st in enumerate(s_t):
        col = R1[:, int(np.clip(round(st - s[0]), 0, len(s) - 1))]
        peak = lvl[int(np.clip(round(st - s[0]), 0, len(s) - 1))]
        above = np.nan_to_num(col) > 0.5 * max(peak, 0.2)
        a = b = i0
        while a > 0 and above[a - 1]:
            a -= 1
        while b < len(t_all) - 1 and above[b + 1]:
            b += 1
        lengths[n] = b - a + 1

    # absolute numbering
    sign = 1 if increases_to == 'right' else -1
    j_at0 = -icpt / slope
    koff_c = scale_mm_approx * 1000.0 / div - sign * j_at0
    W_n = int(np.ceil(search_mm * 1000.0 / div))
    cand = np.arange(int(round(koff_c)) - W_n, int(round(koff_c)) + W_n + 1)
    scores = np.array([_level_score(lengths, profile.level(sign * j + ko)) for ko in cand])
    if np.all(np.isnan(scores)):
        if W_n > 0:
            raise RuntimeError("tick numbering ambiguous: no tick hierarchy in view; "
                               "give scale_mm to better than half a division")
        best, margin, ko = np.nan, np.nan, int(round(koff_c))
    else:
        order = np.argsort(-np.nan_to_num(scores, nan=-2))
        best = scores[order[0]]
        second = scores[order[1]] if len(order) > 1 else -1
        margin = best - np.nan_to_num(second, nan=-1)
        ko = int(cand[order[0]])
        if best < 0.5 or margin < 0.1:
            raise RuntimeError(f"tick numbering ambiguous (best score {best:.2f}, margin "
                               f"{margin:.2f}); check scale_mm and the profile's tick levels")
    k = sign * j + ko
    X = cert.position(k * div)
    use = ~profile.excluded(k * div)
    if not np.all(cert.covers(k[use] * div)):
        flags.append("some ticks outside the certified range")
    s_fit = np.where(use, s_t, np.nan)
    coef, res, keep = _fit_scale(s_fit, X, fit_order)
    coef_lin, _, _ = _fit_scale(np.where(keep, s_t, np.nan), X, 1)
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    if keep.sum() < 0.8 * use.sum():
        flags.append(f"{use.sum() - keep.sum()} ticks rejected in the scale fit")
    return TickResult(polarity, theta, xc, y0, base_rms, band_t, s_t, lengths, k, X, keep,
                      coef, coef_lin, res, rms, float(best), float(margin),
                      int(np.sign(coef[-2])), flags)


def chrome_mask(ink, threshold=0.3, grow_px=3):
    m = ink > threshold
    return ndi.binary_dilation(m, iterations=grow_px)
