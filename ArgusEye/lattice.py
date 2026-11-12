# Argus camera model and lattice fits.
#
# A diffraction order (m, n) leaves the grating with direction cosines
#     (alpha, beta) = (alpha0, beta0) + m * g1 + n * g2,   |g1| = lambda / period_x,  |g2| = lambda / period_y
# which are exactly linear in the orders; (alpha0, beta0) is the zero-order direction (grating tilt).
#
# Camera: pinhole (f*tan(theta)) with Brown-Conrady distortion, in the OpenCV convention:
#     t = (alpha, beta) / gamma,  gamma = sqrt(1 - alpha^2 - beta^2)    normalised image coordinates
#     r^2 = tx^2 + ty^2
#     xd = tx (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 tx ty + p2 (r^2 + 2 tx^2)
#     yd = ty (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 ty^2) + 2 p2 tx ty
#     pixel = principal point + F * (xd, yd)                              (fx = fy = F, no skew)
# Directions in t are measured from the lens axis. The fit finds where the reference's zero order
# lies relative to that axis (the axis offset), which also places the centre of the distortion.
# Which terms are fitted is set in config.ini (distortion = k1, k1 p1 p2, ...). Over the few degrees
# a Fourier lens covers, the higher radial terms (k2, k3) are poorly determined and extrapolate
# badly beyond the reference spots, so k1 is the default.
#
# Reference fit: g1, g2 known (period from config), orthogonal, rotated by phi. Fit F, the principal
#                point, phi, the axis offset and the distortion coefficients.
# Sample fit:    camera fixed from the reference. Fit (alpha0, beta0) relative to the reference's zero
#                order and the free vectors g1, g2 (which absorb rotation and any non-orthogonality).
#                |g| gives the sample period.
#
# The wavelength used here cancels: F calibrated with lambda scales with lambda, and the sample's
# |g| is measured on that same scale, so the sample period comes out in units of the reference
# period and the reported angles are those at the report wavelength.

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from spots import outer_order

T_SCALE = 0.04                 # typical |t|; coefficients are fitted in units scaled by this
FD_REL_STEP = 1e-6
CENTROID_FLOOR_PX = 0.005      # minimum per-spot position error used for weighting
ADEQUATE_RMS_RATIO = 1.5       # an alternative model fitting worse than this (x the main rms) is not counted


@dataclass(frozen=True)
class DistortionModel:
    """Which Brown-Conrady terms are fitted: radial k1..k<n_k>, and p1, p2 if tangential."""
    n_k: int = 1
    tangential: bool = False

    @classmethod
    def parse(cls, text):
        names = text.replace(',', ' ').lower().split()
        ks = sorted(int(n[1]) for n in names if re.fullmatch(r'k[1-3]', n))
        ps = sorted(n for n in names if re.fullmatch(r'p[12]', n))
        if (len(ks) + len(ps) != len(names) or not ks or ks != list(range(1, len(ks) + 1))
                or ps not in ([], ['p1', 'p2'])):
            raise ValueError(f"distortion = '{text}': use the radial terms k1 (k2, k3) in order, "
                             "optionally with p1 p2; e.g. 'k1' or 'k1 p1 p2'")
        return cls(len(ks), bool(ps))

    @property
    def name(self):
        return ' '.join([f'k{i}' for i in range(1, self.n_k + 1)] + (['p1', 'p2'] if self.tangential else []))

    @property
    def n_coef(self):
        return self.n_k + (2 if self.tangential else 0)

    def alternative(self):
        """The model compared against for the distortion-model uncertainty: tangential terms toggled."""
        return DistortionModel(self.n_k, not self.tangential)


def project(dc, F, pp, axis, k, p):
    """Direction cosines (N, 2), relative to the reference zero order, to pixel positions (N, 2)."""
    d = axis + dc
    t = d / np.sqrt(1 - (d ** 2).sum(axis=1, keepdims=True))
    u = t / T_SCALE
    x, y = u[:, 0], u[:, 1]
    r2 = x * x + y * y
    radial = 1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k))
    xd = x * radial + 2 * p[0] * x * y + p[1] * (r2 + 2 * x * x)
    yd = y * radial + p[0] * (r2 + 2 * y * y) + 2 * p[1] * x * y
    return pp + F * T_SCALE * np.c_[xd, yd]


def _rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def _weighted_fit(fun, p0, sig):
    """least_squares on (model - data)/sigma; returns params, covariance (Birge-scaled), chi2_red."""
    fit = least_squares(lambda p: (fun(p) / sig).ravel(), p0, method='lm', x_scale='jac')
    dof = max(fit.fun.size - len(p0), 1)
    chi2_red = float((fit.fun ** 2).sum() / dof)
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) * max(chi2_red, 1.0)
    return fit.x, cov, chi2_red


def _spot_arrays(spots):
    orders = np.array([[s.m, s.n] for s in spots], float)
    pos = np.array([[s.x, s.y] for s in spots])
    sig = np.maximum(np.array([[s.x_err, s.y_err] for s in spots]), CENTROID_FLOOR_PX)
    return orders, pos, sig


@dataclass
class Camera:
    F: float                   # focal length in pixels
    pp: np.ndarray             # principal point (lens axis on the sensor), px
    axis: np.ndarray           # reference zero-order direction relative to the lens axis (dir. cosines)
    k: np.ndarray              # radial coefficients, in fitted (T_SCALE) units
    p: np.ndarray              # tangential p1, p2, in fitted units; zeros when not fitted
    model: DistortionModel

    def project(self, dc):
        return project(dc, self.F, self.pp, self.axis, self.k, self.p)

    @property
    def c(self):
        """Pixel position of the reference zero-order direction."""
        return self.project(np.zeros((1, 2)))[0]

    @property
    def k_opencv(self):
        return self.k / T_SCALE ** (2 * np.arange(1, len(self.k) + 1))

    @property
    def p_opencv(self):
        return self.p / T_SCALE


@dataclass
class ReferenceFit:
    camera: Camera
    phi: float                 # grating rotation on the sensor, rad
    params: np.ndarray         # (F, ppx, ppy, phi, axis_a, axis_b, k..., p1, p2)
    cov: np.ndarray
    residuals: np.ndarray      # (N, 2) model - measured, px
    chi2_red: float
    g1: np.ndarray
    g2: np.ndarray
    loo_rms: float = np.nan    # leave-one-out prediction rms, px
    extrapolation_rms: float = np.nan  # outer ring predicted from a fit to the spots inside it, px
    t_extent: np.ndarray = None  # max |tx|, |ty| covered by the reference spots

    @property
    def model(self):
        return self.camera.model

    @property
    def F(self):
        return self.camera.F

    @property
    def c(self):
        return self.camera.c

    @property
    def u_F(self):
        return float(np.sqrt(self.cov[0, 0]))


def _camera_from_params(p, model):
    i = 6 + model.n_k
    tang = np.asarray(p[i:i + 2]) if model.tangential else np.zeros(2)
    return Camera(F=p[0], pp=np.asarray(p[1:3]), axis=np.asarray(p[4:6]), k=np.asarray(p[6:i]),
                  p=tang, model=model)


def _reference_model(orders, steps, model):
    def fun(p):
        dc = (orders * steps) @ _rot(p[3]).T
        return _camera_from_params(p, model).project(dc)
    return fun


def _fit_reference_spots(orders, pos, sig, steps, model, p0):
    fun = _reference_model(orders, steps, model)
    return _weighted_fit(lambda q: fun(q) - pos, p0, sig)


def fit_reference(pattern, sin_x, sin_y, F0, model=DistortionModel(), leave_one_out=True):
    """
    Calibrates the camera from the reference pattern. sin_x, sin_y are lambda/period of the
    reference along its two axes (direction-cosine step per order). With leave_one_out, also runs
    the leave-one-out and extrapolation checks.
    """
    orders, pos, sig = _spot_arrays(pattern.used)
    steps = np.array([sin_x, sin_y])
    n_par = 6 + model.n_coef
    if orders.shape[0] * 2 <= n_par:
        raise RuntimeError(f"Not enough reference spots for distortion model {model.name}")
    p0 = np.r_[F0, pattern.center, np.arctan2(pattern.a1[1], pattern.a1[0]), 0.0, 0.0, 0.0]
    base = DistortionModel(1, False)
    if model != base:
        # Larger models start from the converged k1 fit; from a cold start they can step off the sphere
        p_base, _, _ = _fit_reference_spots(orders, pos, sig, steps, base, p0)
        p0 = np.r_[p_base, np.zeros(model.n_coef - 1)]
    p, cov, chi2 = _fit_reference_spots(orders, pos, sig, steps, model, p0)

    loo = extrap = np.nan
    if leave_one_out:
        errs = []
        for j in range(len(orders)):
            keep = np.arange(len(orders)) != j
            pj, _, _ = _fit_reference_spots(orders[keep], pos[keep], sig[keep], steps, model, p)
            errs.append(_reference_model(orders[j:j + 1], steps, model)(pj)[0] - pos[j])
        loo = float(np.sqrt((np.array(errs) ** 2).sum(axis=1).mean()))

        # Extrapolation check: calibrate on the spots inside the outer ring, predict the outer ring
        outer = ((np.abs(orders[:, 0]) == np.abs(orders[:, 0]).max())
                 | (np.abs(orders[:, 1]) == np.abs(orders[:, 1]).max()))
        if outer.any() and (~outer).sum() >= n_par:
            pi, _, _ = _fit_reference_spots(orders[~outer], pos[~outer], sig[~outer], steps, model, p)
            e = _reference_model(orders[outer], steps, model)(pi) - pos[outer]
            extrap = float(np.sqrt((e ** 2).sum(axis=1).mean()))

    dc = (orders * steps) @ _rot(p[3]).T
    t = dc / np.sqrt(1 - (dc ** 2).sum(axis=1, keepdims=True))
    R = _rot(p[3])
    return ReferenceFit(camera=_camera_from_params(p, model), phi=p[3], params=p, cov=cov,
                        residuals=_reference_model(orders, steps, model)(p) - pos, chi2_red=chi2,
                        g1=R @ [sin_x, 0.0], g2=R @ [0.0, sin_y], loo_rms=loo,
                        extrapolation_rms=extrap, t_extent=np.abs(t).max(axis=0))


@dataclass
class SampleFit:
    zero_dir: np.ndarray       # (alpha0, beta0), zero-order direction relative to the reference's
    g1: np.ndarray             # direction-cosine step per order along the grid x axis
    g2: np.ndarray             # along the grid y axis
    cov: np.ndarray            # covariance of (alpha0, beta0, g1x, g1y, g2x, g2y), this fit only
    residuals: np.ndarray
    chi2_red: float
    camera: Camera
    u_g_calib: np.ndarray = field(default_factory=lambda: np.zeros(2))   # |g1|,|g2| from the camera
    n_extrapolated: int = 0    # used spots outside the field the reference calibrated
    t_extent: np.ndarray = None  # max |tx|, |ty| reached by the sample spots

    @property
    def c(self):
        """Pixel position of the sample's zero-order direction."""
        return self.camera.project(self.zero_dir[np.newaxis])[0]

    @property
    def mags(self):
        return np.array([np.hypot(*self.g1), np.hypot(*self.g2)])


def _fit_lattice(orders, pos, sig, camera, p0):
    def model(p):
        dc = p[:2] + orders[:, :1] * p[2:4] + orders[:, 1:] * p[4:6]
        return camera.project(dc) - pos
    p, cov, chi2 = _weighted_fit(model, p0, sig)
    return p, cov, chi2, model(p)


def fit_sample(pattern, ref):
    """Measures the sample lattice in direction cosines using the reference camera calibration."""
    orders, pos, sig = _spot_arrays(pattern.used)
    cam = ref.camera
    # Starting lattice: the pixel lattice vectors span one grid step (2 orders for even counts)
    step_x = 2 if pattern.nx % 2 == 0 else 1
    step_y = 2 if pattern.ny % 2 == 0 else 1
    p0 = np.r_[(pattern.center - cam.c) / cam.F, pattern.a1 / (cam.F * step_x),
               pattern.a2 / (cam.F * step_y)]
    p, cov, chi2, res = _fit_lattice(orders, pos, sig, cam, p0)
    fit = SampleFit(zero_dir=p[:2], g1=p[2:4], g2=p[4:6], cov=cov, residuals=res, chi2_red=chi2,
                    camera=cam)

    dc = p[:2] + orders[:, :1] * p[2:4] + orders[:, 1:] * p[4:6]
    t = dc / np.sqrt(1 - (dc ** 2).sum(axis=1, keepdims=True))
    fit.n_extrapolated = int((np.abs(t) > ref.t_extent * 1.02).any(axis=1).sum())
    fit.t_extent = np.abs(t).max(axis=0)

    # Propagate the reference calibration's covariance into |g1|, |g2| by finite differences
    def mags(ref_params):
        q, *_ = _fit_lattice(orders, pos, sig, _camera_from_params(ref_params, cam.model), p)
        return np.array([np.hypot(*q[2:4]), np.hypot(*q[4:6])])
    base = fit.mags
    jac = np.zeros((2, len(ref.params)))
    for k in range(len(ref.params)):
        if k == 3:
            continue           # the reference's rotation does not enter the camera
        dp = np.zeros(len(ref.params))
        dp[k] = FD_REL_STEP * max(abs(ref.params[k]), 1.0)
        jac[:, k] = (mags(ref.params + dp) - base) / dp[k]
    fit.u_g_calib = np.sqrt(np.diag(jac @ ref.cov @ jac.T))
    return fit


def magnitude_uncertainty(g, cov2):
    """Standard uncertainty of |g| from the 2x2 covariance of g."""
    unit = g / np.hypot(*g)
    return float(np.sqrt(unit @ cov2 @ unit))


@dataclass
class AxisResult:
    axis: str
    count: int                 # spots along this axis
    sin_step: float            # lambda / period
    u_rel: float               # relative standard uncertainty of sin_step
    period_um: float
    sep_pm1_deg: float         # angle between orders -1 and +1
    adjacent_deg: float        # angle between neighbouring spots at the centre
    outer_order: int
    full_deg: float            # angle between the outermost spots
    u_sep_pm1_deg: float
    u_adjacent_deg: float
    u_full_deg: float


def axis_result(axis, count, sin_step, u_rel, wavelength_nm):
    """Angles along one grating axis, in the plane containing that axis and the optical axis."""
    def angle(order):
        return 2 * np.degrees(np.arcsin(order * sin_step))

    def u_angle(order):
        return 2 * np.degrees(order * sin_step * u_rel / np.sqrt(1 - (order * sin_step) ** 2))

    m_out = outer_order(count)
    even = count % 2 == 0
    sep, u_sep = angle(1), u_angle(1)
    return AxisResult(axis=axis, count=count, sin_step=sin_step, u_rel=u_rel,
                      period_um=wavelength_nm * 1e-3 / sin_step,
                      sep_pm1_deg=sep, u_sep_pm1_deg=u_sep,
                      adjacent_deg=sep if even else sep / 2,
                      u_adjacent_deg=u_sep if even else u_sep / 2,
                      outer_order=m_out, full_deg=angle(m_out), u_full_deg=u_angle(m_out))
