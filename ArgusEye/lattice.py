# Argus camera model and lattice fits.
#
# A diffraction order (m, n) leaves the grating with direction cosines
#     (alpha, beta) = (alpha0, beta0) + m * g1 + n * g2,   |g1| = lambda / period_x,  |g2| = lambda / period_y
# which are exactly linear in the orders; (alpha0, beta0) is the zero-order direction (grating tilt).
# The Fourier lens and camera map a direction to the sensor at
#     pixel = c + F * D(t),   t = (alpha, beta) / gamma,   gamma = sqrt(1 - alpha^2 - beta^2)
# i.e. an ideal f*tan(theta) lens (F = f / pixel pitch) plus a 2D polynomial distortion
#     D(t) = t + sum over 2 <= i + j <= degree of  a_ij * tx^i * ty^j
# The polynomial has no linear terms, so the x and y scales stay tied together by the single F
# and the reference's known period calibrates both axes. A polynomial (rather than a radial term)
# also covers a beam that passes the lens off-centre or a tilted sensor.
#
# Reference fit: g1, g2 known (period from config), orthogonal, rotated by phi; alpha0 = beta0 = 0,
#                which defines c. Fit F, c, phi and the distortion coefficients.
# Sample fit:    camera fixed from the reference. Fit (alpha0, beta0) and the free vectors g1, g2
#                (which absorb rotation and any non-orthogonality). |g| gives the sample period.
#
# The wavelength used here cancels: F calibrated with lambda scales with lambda, and the sample's
# |g| is measured on that same scale, so the sample period comes out in units of the reference
# period and the reported angles are those at the report wavelength.

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from spots import outer_order

T_SCALE = 0.04                 # typical |t|; keeps the polynomial coefficients near unity
FD_REL_STEP = 1e-6
CENTROID_FLOOR_PX = 0.005      # minimum per-spot position error used for weighting


def distortion_terms(degree):
    return [(i, d - i) for d in range(2, degree + 1) for i in range(d + 1)]


def project(dc, F, c, coef, terms):
    """Direction cosines (N, 2) to pixel positions (N, 2)."""
    gamma = np.sqrt(1 - (dc ** 2).sum(axis=1, keepdims=True))
    t = dc / gamma
    u = t / T_SCALE
    d = t.copy()
    for a, (i, j) in zip(coef, terms):
        d += np.outer(u[:, 0] ** i * u[:, 1] ** j * T_SCALE, a)
    return c + F * d


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
    c: np.ndarray              # pixel position of direction (0, 0)
    coef: np.ndarray           # (n_terms, 2) distortion coefficients
    terms: list

    def project(self, dc):
        return project(dc, self.F, self.c, self.coef, self.terms)


@dataclass
class ReferenceFit:
    camera: Camera
    phi: float                 # grating rotation on the sensor, rad
    degree: int
    params: np.ndarray         # (F, cx, cy, phi, coefficients...)
    cov: np.ndarray
    residuals: np.ndarray      # (N, 2) model - measured, px
    chi2_red: float
    g1: np.ndarray
    g2: np.ndarray
    loo_rms: float = np.nan    # leave-one-out prediction rms, px
    t_extent: np.ndarray = None  # max |tx|, |ty| covered by the reference spots

    @property
    def F(self):
        return self.camera.F

    @property
    def c(self):
        return self.camera.c

    @property
    def u_F(self):
        return float(np.sqrt(self.cov[0, 0]))


def _camera_from_params(p, terms):
    return Camera(F=p[0], c=np.asarray(p[1:3]), coef=np.asarray(p[4:]).reshape(-1, 2), terms=terms)


def _reference_model(orders, steps, terms):
    def model(p):
        dc = (orders * steps) @ _rot(p[3]).T
        return _camera_from_params(p, terms).project(dc)
    return model


def fit_reference(pattern, sin_x, sin_y, F0, degree=3, leave_one_out=True):
    """
    Calibrates the camera from the reference pattern. sin_x, sin_y are lambda/period of the
    reference along its two axes (direction-cosine step per order).
    """
    orders, pos, sig = _spot_arrays(pattern.used)
    steps = np.array([sin_x, sin_y])
    terms = distortion_terms(degree)
    if orders.shape[0] * 2 <= 4 + 2 * len(terms):
        raise RuntimeError(f"Not enough reference spots for distortion degree {degree}")
    model = _reference_model(orders, steps, terms)
    p0 = np.r_[F0, pattern.center, np.arctan2(pattern.a1[1], pattern.a1[0]), np.zeros(2 * len(terms))]
    p, cov, chi2 = _weighted_fit(lambda q: model(q) - pos, p0, sig)

    loo = np.nan
    if leave_one_out:
        errs = []
        for k in range(len(orders)):
            keep = np.arange(len(orders)) != k
            mk = _reference_model(orders[keep], steps, terms)
            pk, _, _ = _weighted_fit(lambda q: mk(q) - pos[keep], p, sig[keep])
            errs.append(_reference_model(orders[k:k + 1], steps, terms)(pk)[0] - pos[k])
        loo = float(np.sqrt((np.array(errs) ** 2).sum(axis=1).mean()))

    dc = (orders * steps) @ _rot(p[3]).T
    t = dc / np.sqrt(1 - (dc ** 2).sum(axis=1, keepdims=True))
    R = _rot(p[3])
    return ReferenceFit(camera=_camera_from_params(p, terms), phi=p[3], degree=degree, params=p,
                        cov=cov, residuals=model(p) - pos, chi2_red=chi2,
                        g1=R @ [sin_x, 0.0], g2=R @ [0.0, sin_y], loo_rms=loo,
                        t_extent=np.abs(t).max(axis=0))


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

    # Propagate the reference calibration's covariance into |g1|, |g2| by finite differences
    def mags(ref_params):
        q, *_ = _fit_lattice(orders, pos, sig, _camera_from_params(ref_params, cam.terms), p)
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
