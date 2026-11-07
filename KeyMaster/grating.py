# Finding and locating the grating lattice in one image.
#
#   1. Prepare the grating signal: normalised image with chrome, saturation and borders masked.
#   2. Coarse lattice: search the 2D power spectrum for the harmonic comb of the nominal lattice
#      over a small range of angle and scale (period from config, um/px from the ticks).
#   3. Refine the strongest harmonic peaks with a zoomed DFT and fit the reciprocal basis G.
#   4. Template: fold the image into one unit cell using G (first image of each pass).
#   5. Coarse position: correlate the image's harmonics with the template's (Fourier domain),
#      over one cell; the scores at half-cell shifts measure how well the image separates the
#      true period from a half period (odd-orders-only gratings).
#   6. Fine position: real-space Gauss-Newton fit of the template to the unmasked pixels,
#      refining the lattice origin p and the basis G together.
#
# Conventions: G has rows g1 (along the scale axis) and g2 (cycles/px). Cell coordinates of a
# pixel r are c = G (r - r0 - p); the template origin (c = 0) sits at r0 + p. Lattice vectors
# (rows of A) satisfy G A^T = I. A 1D grating uses only g1 and a scalar shift.

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize


# ---------------------------------------------------------------- preparation

def prepare(norm, chrome, saturated=None, border_px=24):
    H, W = norm.shape
    w = ~chrome
    if saturated is not None:
        w &= ~ndi.binary_dilation(saturated, iterations=2)
    w[:border_px] = w[-border_px:] = False
    w[:, :border_px] = w[:, -border_px:] = False
    g = norm.astype(np.float64)
    g = g - np.median(g[w])
    g[~w] = 0.0
    return g, w


def hann2d(H, W):
    return np.outer(np.hanning(H), np.hanning(W))


def harmonic_list(hmax, dims, fmax_cpp, G):
    hs = []
    for h1 in range(0, hmax + 1):
        for h2 in (range(-hmax, hmax + 1) if dims == 2 else [0]):
            if h1 == 0 and h2 <= 0:
                continue
            f = h1 * G[0] + (h2 * G[1] if dims == 2 else 0)
            if np.hypot(*f) <= fmax_cpp:
                hs.append((h1, h2))
    return np.array(hs, dtype=int)


def basis(angle, period_along_px, period_perp_px):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c / period_along_px, s / period_along_px],
                     [-s / period_perp_px, c / period_perp_px]])


def lattice_vectors(G, dims):
    if dims == 1:
        g1 = G[0]
        return np.array([g1 / np.dot(g1, g1)])
    return np.linalg.inv(G).T


# ---------------------------------------------------------------- spectra

def zoom_dft(gw, freqs_x, freqs_y, x0=0.0, y0=0.0):
    """Sum gw(y,x) exp(-2 pi i (fx (x-x0) + fy (y-y0))) on the grid freqs_y x freqs_x."""
    H, W = gw.shape
    x = np.arange(W) - x0
    y = np.arange(H) - y0
    Ex = np.exp(-2j * np.pi * np.outer(x, freqs_x)).astype(np.complex64)
    Ey = np.exp(-2j * np.pi * np.outer(freqs_y, y)).astype(np.complex64)
    return Ey @ (gw.astype(np.complex64) @ Ex)


def dft_at(gw, F, x0, y0):
    """Sum gw exp(-2 pi i f.(r - r0)) at each 2D frequency F[n] = (fx, fy)."""
    H, W = gw.shape
    x = np.arange(W) - x0
    y = np.arange(H) - y0
    Ex = np.exp(-2j * np.pi * np.outer(x, F[:, 0]))          # W x N
    M = gw @ Ex                                              # H x N
    Ey = np.exp(-2j * np.pi * np.outer(y, F[:, 1]))          # H x N
    return np.sum(Ey * M, axis=0)


@dataclass
class Coarse:
    G: np.ndarray
    angle: float
    scale: float
    noise: float


def coarse_lattice(g, w, dims, period_along_px, period_perp_px, axis_angle, max_angle_rad,
                   scale_range=0.005, hmax=10):
    H, W = g.shape
    win = hann2d(H, W) * ndi.gaussian_filter(w.astype(np.float32), 2.0)
    F = np.fft.fft2((g * win).astype(np.float32))
    P = (F.real ** 2 + F.imag ** 2).astype(np.float64)
    P = ndi.uniform_filter(P, 3, mode='wrap')
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fr = np.hypot(fx, fy)
    fmax = 0.3
    band = (fr > 0.5 / max(period_along_px, 1)) & (fr < fmax)
    noise = float(np.median(P[band]))

    def score(angle, scale):
        G = basis(axis_angle + angle, period_along_px * scale, period_perp_px * scale)
        hs = harmonic_list(hmax, dims, fmax, G)
        f = hs[:, :1] * G[0] + (hs[:, 1:] * G[1] if dims == 2 else 0)
        coords = [np.mod(f[:, 1] * H, H), np.mod(f[:, 0] * W, W)]
        return np.sum(ndi.map_coordinates(P, coords, order=1, mode='grid-wrap'))

    fmax_h = hmax / period_along_px
    d_ang = 0.25 / (fmax_h * W)
    n_ang = int(np.ceil(max_angle_rad / d_ang))
    angles = np.linspace(-max_angle_rad, max_angle_rad, 2 * n_ang + 1)
    d_sc = 0.25 / (fmax_h * W) * period_along_px / hmax * hmax / period_along_px
    n_sc = int(np.ceil(scale_range / max(d_sc, 1e-5)))
    scales = 1.0 + np.linspace(-scale_range, scale_range, 2 * n_sc + 1)
    best = (-1, 0.0, 1.0)
    for a in angles:
        for sc in scales:
            v = score(a, sc)
            if v > best[0]:
                best = (v, a, sc)
    _, a, sc = best
    G = basis(axis_angle + a, period_along_px * sc, period_perp_px * sc)
    return Coarse(G, a, sc, noise)


def refine_basis(g, w, G, dims, noise, hmax=10, top=40, snr_min=20.0):
    """Zoomed-DFT peak refinement of the strongest harmonics, then LS fit of G."""
    H, W = g.shape
    gw = g * hann2d(H, W) * ndi.gaussian_filter(w.astype(np.float32), 2.0)
    hs = harmonic_list(hmax, dims, 0.3, G)
    f = hs[:, :1] * G[0] + (hs[:, 1:] * G[1] if dims == 2 else 0)
    # power at the predicted harmonics
    amp = np.abs(dft_at(gw.astype(np.complex128), f, 0, 0)) ** 2
    order = np.argsort(-amp)
    sel = [i for i in order[:top] if amp[i] > snr_min * noise]
    if len(sel) < (2 if dims == 1 else 3):
        return G, np.zeros(0, int), amp
    step = 0.5
    meas = []
    for i in sel:
        fx0, fy0 = f[i]
        dx = np.array([-step, 0, step]) / W
        dy = np.array([-step, 0, step]) / H
        C = np.abs(zoom_dft(gw, fx0 + dx, fy0 + dy)) ** 2 + 1e-30
        L = np.log(C)

        def vertex(l, c, r):
            den = l - 2 * c + r
            return 0.0 if den >= 0 else 0.5 * (l - r) / den

        ox = np.clip(vertex(L[1, 0], L[1, 1], L[1, 2]), -1, 1) * step / W
        oy = np.clip(vertex(L[0, 1], L[1, 1], L[2, 1]), -1, 1) * step / H
        meas.append((fx0 + ox, fy0 + oy))
    meas = np.array(meas)
    Hs = hs[sel].astype(float)
    wts = np.sqrt(amp[sel])
    if dims == 1:
        Gn = G.copy()
        h1 = Hs[:, 0]
        for k in range(2):
            Gn[0, k] = np.sum(wts * h1 * meas[:, k]) / np.sum(wts * h1 * h1)
        return Gn, hs[sel], amp
    Gn = np.zeros((2, 2))
    for k in range(2):
        sol, *_ = np.linalg.lstsq(Hs * wts[:, None], meas[:, k] * wts, rcond=None)
        Gn[:, k] = sol
    return Gn, hs[sel], amp


# ---------------------------------------------------------------- template

@dataclass
class Template:
    dims: int
    T: np.ndarray            # cell samples, origin at index 0, zero mean
    Nc: int
    Tf: np.ndarray           # FFT of T
    spl: np.ndarray = None
    d1: np.ndarray = None
    d2: np.ndarray = None
    harmonics: np.ndarray = None   # set used for the coarse correlation
    parity: dict = field(default_factory=dict)

    def finish(self, hmax=10, top=80):
        T = self.T
        if self.dims == 2:
            T = ndi.gaussian_filter(T, 0.6, mode='wrap')
            k = np.fft.fftfreq(self.Nc, 1.0 / self.Nc)
            Tf = np.fft.fft2(T)
            self.d1 = np.real(np.fft.ifft2(Tf * 2j * np.pi * k[None, :]))
            self.d2 = np.real(np.fft.ifft2(Tf * 2j * np.pi * k[:, None]))
            self.d1 = ndi.spline_filter(self.d1, 3, mode='grid-wrap')
            self.d2 = ndi.spline_filter(self.d2, 3, mode='grid-wrap')
            self.spl = ndi.spline_filter(T, 3, mode='grid-wrap')
            self.Tf = np.fft.fft2(self.T)
            hs, mag = [], []
            pe = {'ee': 0.0, 'oe': 0.0, 'eo': 0.0, 'oo': 0.0}
            for h1 in range(0, hmax + 1):
                for h2 in range(-hmax, hmax + 1):
                    if h1 == 0 and h2 <= 0:
                        continue
                    a = abs(self.Tf[h2 % self.Nc, h1 % self.Nc]) ** 2
                    hs.append((h1, h2))
                    mag.append(a)
                    pe[('o' if h1 % 2 else 'e') + ('o' if h2 % 2 else 'e')] += a
        else:
            T = ndi.gaussian_filter1d(T, 0.6, mode='wrap')
            k = np.fft.fftfreq(self.Nc, 1.0 / self.Nc)
            Tf = np.fft.fft(T)
            self.d1 = ndi.spline_filter1d(np.real(np.fft.ifft(Tf * 2j * np.pi * k)), 3,
                                          mode='grid-wrap')
            self.spl = ndi.spline_filter1d(T, 3, mode='grid-wrap')
            self.Tf = np.fft.fft(self.T)
            hs, mag = [], []
            pe = {'e': 0.0, 'o': 0.0}
            for h1 in range(1, 4 * hmax + 1):
                a = abs(self.Tf[h1 % self.Nc]) ** 2
                hs.append((h1, 0))
                mag.append(a)
                pe['o' if h1 % 2 else 'e'] += a
        tot = sum(pe.values()) or 1.0
        self.parity = {k: v / tot for k, v in pe.items()}
        order = np.argsort(-np.array(mag))[:top]
        self.harmonics = np.array(hs)[order]
        return self

    def coef(self, hs):
        if self.dims == 2:
            return self.Tf[hs[:, 1] % self.Nc, hs[:, 0] % self.Nc]
        return self.Tf[hs[:, 0] % self.Nc]


def fold_template(g, w, G, r0, dims, period_px, hmax=10):
    ys, xs = np.nonzero(w)
    d = np.stack([xs - r0[0], ys - r0[1]]).astype(np.float64)
    vals = g[ys, xs]
    if dims == 2:
        Nc = int(np.clip(round(1.5 * period_px), 96, 512))
        c = G @ d
        i = np.rint(np.mod(c[0], 1.0) * Nc).astype(int) % Nc
        j = np.rint(np.mod(c[1], 1.0) * Nc).astype(int) % Nc
        idx = j * Nc + i
        n = Nc * Nc
        shape = (Nc, Nc)
    else:
        Nc = int(np.clip(round(4 * period_px), 128, 4096))
        c = G[0] @ d
        idx = np.rint(np.mod(c, 1.0) * Nc).astype(int) % Nc
        n = Nc
        shape = (Nc,)
    s = np.bincount(idx, weights=vals, minlength=n)
    cnt = np.bincount(idx, minlength=n).astype(float)
    T = np.zeros(n)
    ok = cnt > 0
    T[ok] = s[ok] / cnt[ok]
    T = T.reshape(shape)
    if not ok.all():   # fill empty bins by normalised convolution
        okr = ok.reshape(shape).astype(float)
        num = ndi.gaussian_filter(T * okr, 1.0, mode='wrap')
        den = ndi.gaussian_filter(okr, 1.0, mode='wrap')
        T = np.where(okr > 0, T, num / np.maximum(den, 1e-9))
    T = T - T.mean()
    return Template(dims, T, Nc, None).finish(hmax)


# ---------------------------------------------------------------- coarse position

def coarse_position(g, w, G, r0, tmpl):
    """Best cell offset u (template origin at r0 + G^-1 u) and half-shift score ratios."""
    H, W = g.shape
    gw = g * hann2d(H, W) * ndi.gaussian_filter(w.astype(np.float32), 2.0)
    hs = tmpl.harmonics
    if tmpl.dims == 2:
        F = hs[:, :1] * G[0] + hs[:, 1:] * G[1]
    else:
        F = hs[:, :1] * G[0]
    C = dft_at(gw.astype(np.complex128), F, r0[0], r0[1])
    T = tmpl.coef(hs)
    Q = np.conj(T) * C

    def score(u):
        u = np.atleast_1d(u)
        ph = hs[:, 0] * u[0] + (hs[:, 1] * u[1] if tmpl.dims == 2 else 0)
        return float(np.real(np.sum(Q * np.exp(2j * np.pi * ph))))

    if tmpl.dims == 2:
        grid = np.linspace(0, 1, 64, endpoint=False)
        U1, U2 = np.meshgrid(grid, grid, indexing='ij')
        ph = np.einsum('n,ij->nij', hs[:, 0], U1) + np.einsum('n,ij->nij', hs[:, 1], U2)
        S = np.real(np.einsum('n,nij->ij', Q, np.exp(2j * np.pi * ph)))
        i, j = np.unravel_index(np.argmax(S), S.shape)
        u0 = np.array([grid[i], grid[j]])
        shifts = {'10': (0.5, 0.0), '01': (0.0, 0.5), '11': (0.5, 0.5)}
    else:
        grid = np.linspace(0, 1, 256, endpoint=False)
        S = np.array([score([u]) for u in grid])
        u0 = np.array([grid[np.argmax(S)]])
        shifts = {'10': (0.5,)}

    def refine(u):
        r = optimize.minimize(lambda v: -score(v), u, method='Nelder-Mead',
                              options={'xatol': 1e-6, 'fatol': 1e-12, 'maxiter': 400})
        return r.x, -r.fun

    u_best, s_best = refine(u0)
    ratios = {}
    for name, sh in shifts.items():
        _, s_alt = refine(u_best + np.array(sh))
        ratios[name] = s_alt / s_best if s_best > 0 else np.nan
    u_best = (u_best + 0.5) % 1.0 - 0.5
    return u_best, ratios, s_best


# ---------------------------------------------------------------- fine position

@dataclass
class Fine:
    G: np.ndarray
    p: np.ndarray             # 2D: origin displacement (px) from r0; 1D: shift in cells
    amp: float
    rms: float
    cov_p: np.ndarray
    iters: int
    offset: float = 0.0


def model_rows(shape, y_lo, y_hi, G, p, r0, tmpl, amp, offset):
    """Fitted grating signal (normalised units) on image rows y_lo..y_hi-1."""
    H, W = shape
    yy, xx = np.mgrid[y_lo:y_hi, 0:W].astype(np.float64)
    dx, dy = xx - r0[0], yy - r0[1]
    if tmpl.dims == 2:
        ex, ey = dx - p[0], dy - p[1]
        c1 = G[0, 0] * ex + G[0, 1] * ey
        c2 = G[1, 0] * ex + G[1, 1] * ey
        coords = [np.mod(c2, 1.0) * tmpl.Nc, np.mod(c1, 1.0) * tmpl.Nc]
    else:
        c = G[0, 0] * dx + G[0, 1] * dy - p[0]
        coords = [np.mod(c, 1.0) * tmpl.Nc]
    Tv = ndi.map_coordinates(tmpl.spl, coords, order=3, mode='grid-wrap', prefilter=False)
    return amp * Tv + offset


def fine_fit(g, w, G, p0, r0, tmpl, sub=2, iters=12):
    H, W = g.shape
    mask = np.zeros_like(w)
    mask[::sub, ::sub] = True
    ys, xs = np.nonzero(w & mask)
    vals = g[ys, xs].astype(np.float64)
    dx = xs - r0[0]
    dy = ys - r0[1]
    G = G.copy()
    Nc = tmpl.Nc
    if tmpl.dims == 2:
        p = np.array(p0, float)
        for it in range(iters):
            ex, ey = dx - p[0], dy - p[1]
            c1 = G[0, 0] * ex + G[0, 1] * ey
            c2 = G[1, 0] * ex + G[1, 1] * ey
            coords = [np.mod(c2, 1.0) * Nc, np.mod(c1, 1.0) * Nc]
            Tv = ndi.map_coordinates(tmpl.spl, coords, order=3, mode='grid-wrap', prefilter=False)
            D1 = ndi.map_coordinates(tmpl.d1, coords, order=3, mode='grid-wrap', prefilter=False)
            D2 = ndi.map_coordinates(tmpl.d2, coords, order=3, mode='grid-wrap', prefilter=False)
            A = np.vstack([Tv, np.ones_like(Tv)]).T
            (a, b), *_ = np.linalg.lstsq(A, vals, rcond=None)
            r = vals - a * Tv - b
            J = np.column_stack([
                -a * (D1 * G[0, 0] + D2 * G[1, 0]),
                -a * (D1 * G[0, 1] + D2 * G[1, 1]),
                a * D1 * ex, a * D1 * ey, a * D2 * ex, a * D2 * ey,
                Tv, np.ones_like(Tv)])
            # minus sign: model m = a T(c) + b, r = vals - m, so the step solves J d = r
            nrm = np.linalg.norm(J, axis=0) + 1e-30
            d, *_ = np.linalg.lstsq(J / nrm, r, rcond=None)
            d = d / nrm
            p += d[:2]
            G += np.array([[d[2], d[3]], [d[4], d[5]]])
            if np.max(np.abs(d[:2])) < 1e-5:
                break
        sig2 = np.sum(r ** 2) / max(len(r) - 8, 1)
        JtJ = (J / nrm).T @ (J / nrm)
        cov = np.linalg.pinv(JtJ) / np.outer(nrm, nrm) * sig2
        return Fine(G, p, float(a), float(np.sqrt(sig2)), cov[:2, :2], it + 1, float(b))
    # 1D: c = g1 . (r - r0) - delta
    delta = float(p0)
    for it in range(iters):
        c = G[0, 0] * dx + G[0, 1] * dy - delta
        coords = [np.mod(c, 1.0) * Nc]
        Tv = ndi.map_coordinates(tmpl.spl, coords, order=3, mode='grid-wrap', prefilter=False)
        D1 = ndi.map_coordinates(tmpl.d1, coords, order=3, mode='grid-wrap', prefilter=False)
        A = np.vstack([Tv, np.ones_like(Tv)]).T
        (a, b), *_ = np.linalg.lstsq(A, vals, rcond=None)
        r = vals - a * Tv - b
        J = np.column_stack([-a * D1, a * D1 * dx, a * D1 * dy, Tv, np.ones_like(Tv)])
        nrm = np.linalg.norm(J, axis=0) + 1e-30
        d, *_ = np.linalg.lstsq(J / nrm, r, rcond=None)
        d = d / nrm
        delta += d[0]
        G[0] += d[1:3]
        if abs(d[0]) < 1e-7:
            break
    sig2 = np.sum(r ** 2) / max(len(r) - 5, 1)
    JtJ = (J / nrm).T @ (J / nrm)
    cov = np.linalg.pinv(JtJ) / np.outer(nrm, nrm) * sig2
    return Fine(G, np.array([delta]), float(a), float(np.sqrt(sig2)), cov[:1, :1], it + 1,
                float(b))


# ---------------------------------------------------------------- one image

@dataclass
class GratingResult:
    dims: int
    G: np.ndarray
    A: np.ndarray
    r0: np.ndarray
    p: np.ndarray
    cov_p: np.ndarray
    ratios: dict
    corr: float
    fine_rms: float
    amp: float
    n_harm: int
    coarse_angle: float
    coarse_scale: float
    flags: list = field(default_factory=list)
    offset: float = 0.0


def measure(g, w, r0, dims, period_along_px, period_perp_px, axis_angle, max_angle_rad,
            tmpl=None, hmax=10, scale_range=0.005):
    """Lattice and template-origin position in one image. Builds a template if none given."""
    flags = []
    co = coarse_lattice(g, w, dims, period_along_px, period_perp_px, axis_angle, max_angle_rad,
                        scale_range, hmax)
    if abs(abs(co.angle) - max_angle_rad) < 1e-9:
        flags.append("lattice angle at the search limit")
    if abs(abs(co.scale - 1) - scale_range) < 1e-9:
        flags.append("lattice period at the search limit (check nominal period / um per px)")
    G, used, _ = refine_basis(g, w, co.G, dims, co.noise, hmax)
    if len(used) == 0:
        flags.append("weak grating signal: harmonic refinement skipped")
    build = tmpl is None
    if build:
        tmpl = fold_template(g, w, G, r0, dims, period_along_px, hmax)
        u0 = np.zeros(2 if dims == 2 else 1)
        ratios, corr = {}, np.nan
    else:
        u0, ratios, corr = coarse_position(g, w, G, r0, tmpl)
    if dims == 2:
        p0 = np.linalg.solve(G, u0)
        fine = fine_fit(g, w, G, p0, r0, tmpl)
    else:
        fine = fine_fit(g, w, G, u0[0], r0, tmpl)
    if build:
        # refold with the refined lattice, then fit again so the origin is self-consistent
        tmpl = fold_template(g, w, fine.G, r0 + (fine.p if dims == 2 else 0), dims,
                             period_along_px, hmax) if dims == 2 else \
            fold_template(g, w, fine.G, r0 + fine.p[0] * lattice_vectors(fine.G, 1)[0], dims,
                          period_along_px, hmax)
        u0, ratios, corr = coarse_position(g, w, fine.G, r0, tmpl)
        if dims == 2:
            fine = fine_fit(g, w, fine.G, np.linalg.solve(fine.G, u0), r0, tmpl)
        else:
            fine = fine_fit(g, w, fine.G, u0[0], r0, tmpl)
    res = GratingResult(dims, fine.G, lattice_vectors(fine.G, dims), np.asarray(r0, float),
                        fine.p, fine.cov_p, ratios, corr, fine.rms, fine.amp,
                        len(used), co.angle, co.scale, flags, fine.offset)
    return res, tmpl
