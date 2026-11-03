# Argus spot finding: from a camera frame to a list of indexed, sub-pixel spot centroids.
#
#   1. Estimate the noise, replace isolated hot pixels, and map the slowly varying background.
#   2. Detect spots: smooth, threshold at background + k*sigma, label connected blobs.
#   3. Index the blobs to the expected nx x ny grid of diffraction orders. Lattice vectors come
#      from nearest-neighbour displacements, so camera roll does not matter; the grid window is
#      the nx x ny block with the most light, which rejects faint higher orders and ghosts.
#   4. Refine every spot with a windowed (Gaussian-weighted) centroid on a local plane-fit
#      background, and with a 2D Gaussian fit as a cross-check. Spots that are saturated or cut
#      by the sensor edge are flagged and kept out of the fits.

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HOT_PIXEL_SIGMA = 8.0
BACKGROUND_BLOCK_PX = 64
DETECT_SMOOTH_PX = 2.0
FRACTIONAL_INDEX_LIMIT = 0.25      # max distance from a lattice node, in lattice units
WINDOW_FLUX_TIE = 0.98             # grid windows within 2% of the brightest count as tied
APERTURE_SIGMAS = 4.0              # centroid aperture radius in spot sigmas
CENTROID_TOL_PX = 1e-5
CENTROID_MAX_ITER = 100


@dataclass
class Spot:
    i: int                     # column index within the nx x ny grid (0..nx-1)
    j: int                     # row index (0..ny-1)
    m: int                     # diffraction order along the grid's x axis
    n: int                     # diffraction order along the grid's y axis
    x: float = np.nan          # windowed centroid, px (column)
    y: float = np.nan          # windowed centroid, px (row)
    x_err: float = np.nan
    y_err: float = np.nan
    x_gauss: float = np.nan    # 2D Gaussian fit centre, px
    y_gauss: float = np.nan
    sigma_px: float = np.nan   # Gaussian rms radius, px
    peak: float = np.nan       # background-subtracted peak, counts
    flux: float = np.nan       # background-subtracted sum in the aperture, counts
    snr: float = np.nan        # peak / pixel noise
    saturated: bool = False
    clipped: bool = False
    used: bool = True
    note: str = ''


@dataclass
class Pattern:
    """Everything found in one image."""
    nx: int
    ny: int
    spots: list
    noise: float               # pixel noise sigma, counts
    spot_sigma: float          # typical spot rms radius, px
    a1: np.ndarray             # lattice vector between neighbouring columns, px
    a2: np.ndarray             # lattice vector between neighbouring rows, px
    center: np.ndarray         # grid centre estimate, px
    hot_pixels: int
    n_candidates: int
    missing: list = field(default_factory=list)   # (m, n) orders not found on the sensor
    notes: list = field(default_factory=list)
    image: np.ndarray = None   # hot-pixel-cleaned image
    resid: np.ndarray = None   # image minus the background map

    @property
    def used(self):
        return [s for s in self.spots if s.used]


def order_numbers(count):
    """
    Diffraction orders of a row of `count` spots. An odd count includes the zero order
    (-2..2 for 5 spots); an even count has no zero order and uses odd orders (-5..5 for 6 spots).
    """
    step = 2 if count % 2 == 0 else 1
    return [int(round(step * (k - (count - 1) / 2))) for k in range(count)]


def outer_order(count):
    return order_numbers(count)[-1]


# --- Image statistics -----------------------------------------------------------------------

def estimate_noise(img):
    """Pixel noise from neighbour differences, sigma-clipped so spots and hot pixels don't count."""
    d = np.diff(img, axis=1).ravel()
    for _ in range(5):
        s = d.std()
        d = d[np.abs(d - d.mean()) < 4 * s]
    return max(d.std() / np.sqrt(2), 1e-6)


def clean_hot_pixels(img, noise):
    """Replaces single pixels that stand far above (or below) their 3x3 median."""
    med = ndimage.median_filter(img, size=3)
    hot = np.abs(img - med) > HOT_PIXEL_SIGMA * noise
    # A defect stands alone: its 8 neighbours are flat. The peak of a sharp spot also stands
    # above its 3x3 median, but its neighbours slope steeply, so it is left alone.
    ring = np.ones((3, 3), bool)
    ring[1, 1] = False
    spread = (ndimage.maximum_filter(img, footprint=ring) - ndimage.minimum_filter(img, footprint=ring))
    hot &= spread < 0.5 * HOT_PIXEL_SIGMA * noise
    out = img.copy()
    out[hot] = med[hot]
    return out, int(hot.sum())


def _block_background(img, mask, block):
    """Median of the unmasked pixels in each block, interpolated back to full size."""
    rows, cols = img.shape
    nby, nbx = -(-rows // block), -(-cols // block)
    grid = np.full((nby, nbx), np.nan)
    for by in range(nby):
        for bx in range(nbx):
            sl = (slice(by * block, (by + 1) * block), slice(bx * block, (bx + 1) * block))
            vals = img[sl][~mask[sl]]
            if vals.size > 0.2 * img[sl].size:
                grid[by, bx] = np.median(vals)
    if np.isnan(grid).all():
        return np.full(img.shape, np.median(img))
    # Fill blocks that were mostly spot with the nearest valid block
    idx = ndimage.distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True)
    grid = grid[tuple(idx)]
    grid = ndimage.median_filter(grid, size=3, mode='nearest')
    zoom = (rows / nby, cols / nbx)
    full = ndimage.zoom(grid, zoom, order=1, mode='nearest', grid_mode=True)
    return full[:rows, :cols]


def estimate_background(img, noise, threshold_sigma):
    """Large-scale background with the spots masked out (two passes)."""
    bg = _block_background(img, np.zeros(img.shape, bool), BACKGROUND_BLOCK_PX)
    smooth = ndimage.gaussian_filter(img - bg, DETECT_SMOOTH_PX)
    mask = smooth > 0.5 * threshold_sigma * noise * _smooth_noise_factor()
    mask = ndimage.binary_dilation(mask, iterations=6)
    return _block_background(img, mask, BACKGROUND_BLOCK_PX)


def _smooth_noise_factor():
    """Noise reduction of white noise under the detection Gaussian filter."""
    return 1 / (2 * np.sqrt(np.pi) * DETECT_SMOOTH_PX)


# --- Detection and indexing -----------------------------------------------------------------

def detect_candidates(resid, noise, threshold_sigma, min_area):
    """Connected blobs above threshold in the smoothed, background-subtracted image."""
    smooth = ndimage.gaussian_filter(resid, DETECT_SMOOTH_PX)
    labels, count = ndimage.label(smooth > threshold_sigma * noise * _smooth_noise_factor())
    if count == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    index = np.arange(1, count + 1)
    area = ndimage.sum_labels(np.ones_like(resid), labels, index)
    weights = np.clip(resid, 0, None)
    flux = ndimage.sum_labels(weights, labels, index)
    com = np.array(ndimage.center_of_mass(weights, labels, index))     # (row, col)
    # Second moment of each blob, for the typical spot size
    rr, cc = np.indices(resid.shape)
    m2 = ndimage.sum_labels(weights * (rr ** 2 + cc ** 2), labels, index) / np.maximum(flux, 1e-12)
    radius2 = m2 - (com ** 2).sum(axis=1)
    keep = (area >= min_area) & (flux > 0)
    return com[keep][:, ::-1], flux[keep], np.sqrt(np.clip(radius2[keep] / 2, 0, None))


def _lattice_vectors(pts):
    """Shortest near-horizontal and near-vertical neighbour vectors, medianed over all points."""
    k = min(9, len(pts))
    _, nbr = cKDTree(pts).query(pts, k=k)
    horiz, vert = [], []
    for p, idx in zip(pts, nbr):
        v = pts[idx[1:]] - p
        h = v[np.abs(v[:, 0]) > np.abs(v[:, 1])]
        u = v[np.abs(v[:, 0]) <= np.abs(v[:, 1])]
        if len(h):
            h = h * np.sign(h[:, :1])
            horiz.append(h[np.argmin(np.hypot(*h.T))])
        if len(u):
            u = u * np.sign(u[:, 1:])
            vert.append(u[np.argmin(np.hypot(*u.T))])
    if not horiz or not vert:
        raise RuntimeError("Could not find a 2D lattice of spots")
    return np.median(horiz, axis=0), np.median(vert, axis=0)


def index_grid(pts, flux, nx, ny, shape):
    """
    Assigns each candidate a lattice node and picks the nx x ny block of nodes holding the most
    light. Returns {(i, j): candidate index}, the lattice vectors and the grid centre.
    """
    order = np.argsort(flux)[::-1]
    bright = pts[order[:max(nx * ny, 4)]]
    a1, a2 = _lattice_vectors(bright)
    origin = pts[order[0]]

    for _ in range(2):
        basis = np.column_stack([a1, a2])
        lat = np.linalg.solve(basis, (pts - origin).T).T
        node = np.round(lat).astype(int)
        on_node = np.all(np.abs(lat - node) < FRACTIONAL_INDEX_LIMIT, axis=1)
        if on_node.sum() < 3:
            raise RuntimeError("Too few spots sit on a regular lattice")
        # Refine origin and lattice vectors by least squares over the on-node spots
        design = np.column_stack([np.ones(on_node.sum()), node[on_node]])
        coef, *_ = np.linalg.lstsq(design, pts[on_node], rcond=None)
        origin, a1, a2 = coef[0], coef[1], coef[2]

    # Keep the brightest candidate on each node
    nodes = {}
    for c in np.flatnonzero(on_node):
        key = tuple(node[c])
        if key not in nodes or flux[c] > flux[nodes[key]]:
            nodes[key] = c

    keys = np.array(list(nodes))
    node_flux = np.array([flux[c] for c in nodes.values()])
    image_center = np.array([shape[1] - 1, shape[0] - 1]) / 2
    windows = []
    for i0 in range(keys[:, 0].min() - nx + 1, keys[:, 0].max() + 1):
        for j0 in range(keys[:, 1].min() - ny + 1, keys[:, 1].max() + 1):
            inside = ((keys[:, 0] >= i0) & (keys[:, 0] < i0 + nx)
                      & (keys[:, 1] >= j0) & (keys[:, 1] < j0 + ny))
            mid = origin + (i0 + (nx - 1) / 2) * a1 + (j0 + (ny - 1) / 2) * a2
            windows.append((node_flux[inside].sum(), np.hypot(*(mid - image_center)), i0, j0, mid))
    # Most light wins; when whole rows/columns are off the sensor, several windows tie and the
    # one centred nearest the middle of the image is taken
    brightest = max(w[0] for w in windows)
    _, _, i0, j0, center = min((w for w in windows if w[0] >= WINDOW_FLUX_TIE * brightest),
                               key=lambda w: w[1])

    grid = {}
    for (ki, kj), c in nodes.items():
        if i0 <= ki < i0 + nx and j0 <= kj < j0 + ny:
            grid[(ki - i0, kj - j0)] = c
    return grid, a1, a2, center


# --- Centroiding ----------------------------------------------------------------------------

def _patch(img, x, y, half):
    r0, r1 = int(np.floor(y - half)), int(np.ceil(y + half)) + 1
    c0, c1 = int(np.floor(x - half)), int(np.ceil(x + half)) + 1
    r0c, c0c = max(r0, 0), max(c0, 0)
    r1c, c1c = min(r1, img.shape[0]), min(c1, img.shape[1])
    yy, xx = np.mgrid[r0c:r1c, c0c:c1c]
    return img[r0c:r1c, c0c:c1c], xx.astype(float), yy.astype(float)


def aperture_radii(s, pitch):
    """Centroid aperture and background ring radii, kept clear of the neighbouring spots."""
    r_ap = min(APERTURE_SIGMAS * s, 0.4 * pitch)
    r_in = r_ap + 3
    r_out = min(r_in + max(6.0, 2 * s), pitch - 3.5 * s)
    return r_ap, r_in, max(r_out, r_in + 4)


def windowed_centroid(img, x0, y0, s, noise, pitch):
    """
    Iterative Gaussian-weighted centroid (weight sigma = spot sigma) on a plane-fit local
    background. The update x += 2*<w I dx>/<w I> lands on the centre of a symmetric spot.
    Returns x, y, x_err, y_err, peak, flux, clipped.
    """
    r_ap, r_in, r_out = aperture_radii(s, pitch)
    data, xx, yy = _patch(img, x0, y0, r_out + 1)
    rows, cols = img.shape
    clipped = (x0 - r_ap < 0 or y0 - r_ap < 0 or x0 + r_ap > cols - 1 or y0 + r_ap > rows - 1)

    r = np.hypot(xx - x0, yy - y0)
    ring = (r >= r_in) & (r <= r_out)
    if ring.sum() >= 10:
        design = np.column_stack([np.ones(ring.sum()), xx[ring] - x0, yy[ring] - y0])
        plane, *_ = np.linalg.lstsq(design, data[ring], rcond=None)
        data = data - (plane[0] + plane[1] * (xx - x0) + plane[2] * (yy - y0))

    x, y = x0, y0
    for _ in range(CENTROID_MAX_ITER):
        dx, dy = xx - x, yy - y
        r2 = dx ** 2 + dy ** 2
        w = np.exp(-r2 / (2 * s ** 2)) * (r2 <= r_ap ** 2)
        wsum = (w * data).sum()
        if wsum <= 0:
            return (np.nan,) * 6 + (clipped,)
        step_x = 2 * (w * data * dx).sum() / wsum
        step_y = 2 * (w * data * dy).sum() / wsum
        x, y = x + step_x, y + step_y
        if np.hypot(step_x, step_y) < CENTROID_TOL_PX:
            break

    dx, dy = xx - x, yy - y
    r2 = dx ** 2 + dy ** 2
    w = np.exp(-r2 / (2 * s ** 2)) * (r2 <= r_ap ** 2)
    wsum = (w * data).sum()
    x_err = 2 * noise * np.sqrt(((w * dx) ** 2).sum()) / wsum
    y_err = 2 * noise * np.sqrt(((w * dy) ** 2).sum()) / wsum
    inside = r2 <= r_ap ** 2
    peak = data[inside].max() if inside.any() else np.nan
    flux = data[inside].sum()
    return x, y, x_err, y_err, peak, flux, clipped


def gaussian_fit(img, x0, y0, s, pitch):
    """2D elliptical (axis-aligned) Gaussian plus constant, fitted inside the aperture."""
    r_ap = aperture_radii(s, pitch)[0]
    data, xx, yy = _patch(img, x0, y0, r_ap)
    inside = np.hypot(xx - x0, yy - y0) <= r_ap
    d, x, y = data[inside], xx[inside], yy[inside]
    p0 = [d.max() - np.median(d), x0, y0, s, s, min(np.median(d), d.min())]

    def model(p):
        return p[0] * np.exp(-((x - p[1]) ** 2 / (2 * p[3] ** 2)
                               + (y - p[2]) ** 2 / (2 * p[4] ** 2))) + p[5]

    try:
        fit = least_squares(lambda p: model(p) - d, p0, method='lm')
    except Exception:
        return np.nan, np.nan, np.nan
    return fit.x[1], fit.x[2], np.sqrt(abs(fit.x[3] * fit.x[4]))


def measure_point(img, x0, y0, s, noise, pitch):
    """Finds and centroids the brightest feature within a quarter pitch of (x0, y0)."""
    search_radius = 0.25 * pitch
    data, xx, yy = _patch(img, x0, y0, search_radius)
    smooth = ndimage.gaussian_filter(data, DETECT_SMOOTH_PX)
    inside = np.hypot(xx - x0, yy - y0) <= search_radius
    k = np.argmax(np.where(inside, smooth, -np.inf))
    x, y, x_err, y_err, peak, flux, clipped = windowed_centroid(img, xx.flat[k], yy.flat[k], s,
                                                                noise, pitch)
    return dict(x=x, y=y, x_err=x_err, y_err=y_err, peak=peak, snr=peak / noise)


# --- Top level ------------------------------------------------------------------------------

def measure_pattern(frame, nx, ny, threshold_sigma=8.0, min_area=20, saturation_fraction=0.98):
    """Finds, indexes and centroids the nx x ny spot grid in a Frame."""
    raw = frame.image
    noise = estimate_noise(raw)
    img, hot = clean_hot_pixels(raw, noise)
    bg = estimate_background(img, noise, threshold_sigma)
    resid = img - bg

    pts, flux, radius = detect_candidates(resid, noise, threshold_sigma, min_area)
    if len(pts) < 3:
        raise RuntimeError(f"Only {len(pts)} spots detected; check exposure and threshold")
    grid, a1, a2, center = index_grid(pts, flux, nx, ny, img.shape)

    # Typical spot size from the indexed blobs
    in_grid = np.array(list(grid.values()))
    spot_sigma = float(np.median(radius[in_grid]))

    pitch = min(np.hypot(*a1), np.hypot(*a2))
    xs, ys = order_numbers(nx), order_numbers(ny)
    sat_level = saturation_fraction * frame.full_scale
    spots = []
    for (i, j), c in sorted(grid.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        spot = Spot(i=i, j=j, m=xs[i], n=ys[j])
        x0, y0 = pts[c]
        spot.x, spot.y, spot.x_err, spot.y_err, spot.peak, spot.flux, spot.clipped = \
            windowed_centroid(img, x0, y0, spot_sigma, noise, pitch)
        spot.snr = spot.peak / noise
        if not spot.clipped and np.isfinite(spot.x):
            spot.x_gauss, spot.y_gauss, spot.sigma_px = gaussian_fit(resid, spot.x, spot.y,
                                                                     spot_sigma, pitch)
        patch, _, _ = _patch(raw, x0, y0, aperture_radii(spot_sigma, pitch)[0])
        spot.saturated = bool(patch.max() >= sat_level)

        if spot.saturated:
            spot.used, spot.note = False, 'saturated'
        elif spot.clipped:
            spot.used, spot.note = False, 'cut by sensor edge'
        elif not np.isfinite(spot.x):
            spot.used, spot.note = False, 'centroid failed'
        spots.append(spot)

    # Use the Gaussian-fit size for the reported spot sigma when available
    sigmas = [s.sigma_px for s in spots if s.used and np.isfinite(s.sigma_px)]
    if sigmas:
        spot_sigma = float(np.median(sigmas))

    pattern = Pattern(nx=nx, ny=ny, spots=spots, noise=noise, spot_sigma=spot_sigma,
                      a1=a1, a2=a2, center=center, hot_pixels=hot, n_candidates=len(pts),
                      image=img, resid=resid)
    pattern.missing = [(xs[i], ys[j]) for j in range(ny) for i in range(nx) if (i, j) not in grid]
    if pattern.missing:
        pattern.notes.append(f"{len(pattern.missing)} spots not found (off the sensor or too faint)")
    return pattern
