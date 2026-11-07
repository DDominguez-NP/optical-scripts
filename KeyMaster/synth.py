# Synthetic KeyMaster runs with a known answer.
#
# Renders Keyence-like images of a certified scale (chrome side down, mirrored numerals)
# stacked over a grating, plus run.csv, cert.csv, config.ini and truth.json, so KeyMaster can be
# checked end to end. Modelled: certified tick deviations (piecewise linear between certified
# points), scale and grating thermal expansion, scale tilt, grating angle, radial lens
# distortion, a different magnification for the grating plane (gap parallax), per-image
# magnification jitter, stage jitter, blur, defocus edge contrast, vignetting and shot noise.
#
# Grating types: odd2d (odd orders only on both axes, e.g. 6x8), all2d (all orders, e.g. 9x5),
# mixed2d (odd along x, all along y), odd1d, all1d.
#
#   python synth.py OUTDIR [--profile sm4] [--grating all2d] [--passes fwd:0,rev:0,fwd:180]
#                          [--stops 10] [--no-noise] [--workers 4] [--seed 1]

import os
import csv
import json
import argparse
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi

import scales

TICK_LENGTHS_UM = {   # rank -> total length (assumed layouts, matching scales.PROFILES)
    'edmund59276_metric': {1: 60, 2: 100, 3: 150, 4: 200},
    'edmund59276_english': {1: 60, 2: 100, 3: 150, 4: 200},
    'sm4': {1: 40, 2: 60, 3: 90, 4: 130},
}


@dataclass
class Synth:
    profile: str = 'edmund59276_metric'
    increases_to: str = 'right'
    grating: str = 'odd2d'
    period_x_um: float = 242.1347       # true, at 20 C
    period_y_um: float = 242.0991
    nominal_um: float = 242.12
    feature_um: float = 16.0
    alpha_rad: float = 1.8e-3           # grating lattice vs scale axis
    grating_centre_um: tuple = (3.7, -2.1)
    width: int = 2880
    height: int = 2160
    mpp: float = 1.0400
    mpp_jitter_ppm: float = 40.0
    parallax_ppm: float = 1500.0
    distortion_k1: float = 1.0e-10      # radial, per px^2
    theta_deg: float = 0.35
    stage_jitter_um: float = 50.0
    baseline_offset_um: float = -150.0  # scale coordinate v seen at the image centre
    step_periods: int = 10
    n_stops: int = 10
    passes: list = field(default_factory=lambda: [('fwd', 0), ('rev', 0)])
    temps_c: list = field(default_factory=lambda: [21.30, 21.42, 21.55, 21.38])
    scale_cte: float = 9.0
    grating_cte: float = 0.55
    cert_temp_c: float = 20.0
    cert_step_mm: float = 1.0
    cert_dev_um: float = 0.15
    cert_U_um: float = 0.415
    chrome_T: float = 0.03
    psf_um: float = 1.0
    defocus_um: float = 1.6
    edge_loss: float = 0.10
    edge_defocus: float = 0.08
    signal: float = 28000.0
    read_noise: float = 5.0
    noise: bool = True
    scale_mm_error: float = 0.03
    seed: int = 1


# ---------------------------------------------------------------- grating cell

def make_cell(cfg, N=1024):
    rng = np.random.default_rng(cfg.seed + 1000)
    m = np.fft.fftfreq(N, 1.0 / N)
    k0 = 1.0 / cfg.feature_um
    if cfg.grating.endswith('1d'):
        kmag = np.abs(m) / cfg.period_x_um
        env = np.exp(-0.5 * ((kmag - k0) / (0.35 * k0)) ** 2)
        sel = (np.abs(m) % 2 == 1) if cfg.grating == 'odd1d' else (m != 0)
        F = env * sel * (rng.normal(size=N) + 1j * rng.normal(size=N))
        f = np.real(np.fft.ifft(F))
        fx = np.real(np.fft.ifft(F * 2j * np.pi * m / cfg.period_x_um))
        sc = f.std()
        return f / sc, fx / sc, None
    M, Nn = np.meshgrid(m, m)
    kmag = np.hypot(M / cfg.period_x_um, Nn / cfg.period_y_um)
    env = np.exp(-0.5 * ((kmag - k0) / (0.28 * k0)) ** 2)
    if cfg.grating == 'odd2d':
        sel = (np.abs(M) % 2 == 1) & (np.abs(Nn) % 2 == 1)
    elif cfg.grating == 'mixed2d':
        sel = (np.abs(M) % 2 == 1)
    else:
        sel = ~((M == 0) & (Nn == 0))
    F = env * sel * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    f = np.real(np.fft.ifft2(F))
    fx = np.real(np.fft.ifft2(F * 2j * np.pi * M / cfg.period_x_um))
    fy = np.real(np.fft.ifft2(F * 2j * np.pi * Nn / cfg.period_y_um))
    sc = f.std()
    return f / sc, fx / sc, fy / sc


# ---------------------------------------------------------------- certificate (truth)

def make_cert(cfg, prof):
    rng = np.random.default_rng(cfg.seed + 2000)
    lo, hi = prof.range_mm
    aim = np.arange(np.ceil(lo / cfg.cert_step_mm), np.floor(hi / cfg.cert_step_mm) + 1) \
        * cfg.cert_step_mm
    dev = rng.uniform(-cfg.cert_dev_um, cfg.cert_dev_um, len(aim))
    dev -= np.interp(0.0, aim, dev) if lo < 0 < hi else dev[0]
    return aim, dev


# ---------------------------------------------------------------- one image

def render(cfg, prof, cell, cert_aim, cert_dev, stop):
    rng = np.random.default_rng(stop['seed'])
    W, H = cfg.width, cfg.height
    cx, cy = (W - 1) / 2, (H - 1) / 2
    mpp = cfg.mpp * (1 + stop['mpp_jitter'])
    mpp_g = mpp * (1 + cfg.parallax_ppm * 1e-6)
    th = np.radians(cfg.theta_deg)
    sig = 1 if cfg.increases_to == 'right' else -1
    T = stop['temp_c']
    exp_s = 1 + cfg.scale_cte * 1e-6 * (T - cfg.cert_temp_c)
    exp_g = 1 + cfg.grating_cte * 1e-6 * (T - 20.0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    f = 1 + cfg.distortion_k1 * (dx * dx + dy * dy)
    dx, dy = dx * f, dy * f
    ox, oy = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
    Xc, Yc = stop['Xc'], stop['Yc']
    u = Xc + sig * ox * mpp            # physical position along the scale at T (um)
    v = Yc + oy * mpp                  # across the scale (down in the image)
    ug = Xc + sig * ox * mpp_g
    vg = Yc + oy * mpp_g

    # --- chrome: ticks at certified positions, expanded to T
    div = prof.division_um
    Xr = u / exp_s                     # certificate-temperature coordinate
    x = Xr.copy()
    for _ in range(3):
        x = Xr - np.interp(x, cert_aim * 1000, cert_dev)
    k = np.rint(x / div)
    du = (x - k * div) * exp_s
    soft = lambda d: np.clip(d / mpp + 0.5, 0.0, 1.0)
    lens = TICK_LENGTHS_UM[prof.name]
    len_lut = np.array([0.0] + [lens[r] for r in range(1, 5)])
    L = len_lut[prof.level(k.astype(np.int64))]
    vs = -v                            # face down: mirrored about the long axis
    w2 = prof.line_width_um / 2
    if prof.tick_style == 'symmetric':
        along = soft(L / 2 - np.abs(vs))
    else:
        along = soft(np.minimum(vs, L - vs))
    chrome = np.maximum(soft(w2 - np.abs(du)) * along, soft(w2 - np.abs(vs)))
    if prof.name.startswith('edmund'):
        # the cross: Y-axis line through 0 and its ticks
        x0 = 0.0 + np.interp(0.0, cert_aim * 1000, cert_dev)
        ux = u / exp_s - x0
        kv = np.rint(vs / div)
        Lv = len_lut[prof.level(kv.astype(np.int64))]
        ytick = soft(w2 - np.abs(vs - kv * div)) * soft(Lv / 2 - np.abs(ux))
        chrome = np.maximum(chrome, np.maximum(soft(w2 - np.abs(ux)), ytick))
    # numerals (mirrored)
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    hpx = int(round(55 / mpp))
    try:
        font = ImageFont.truetype("arial.ttf", hpx)
    except OSError:
        font = ImageFont.load_default(size=hpx)
    label = Image.new("L", (W, H), 0)
    kmin, kmax = int(np.nanmin(k)), int(np.nanmax(k))
    for kk in range(kmin, kmax + 1):
        if kk % prof.label_every or kk == 0:
            continue
        val = abs(kk * div / 1000)
        text = f"{val:.1f}" if prof.name.startswith('edmund') else f"{val:.0f}"
        patch = Image.new("L", (320, 140), 0)
        ImageDraw.Draw(patch).text((160, 70), text, fill=255, font=font, anchor="mm")
        patch = ImageOps.flip(patch).rotate(-np.degrees(th), resample=Image.BICUBIC)
        uk = (kk * div + np.interp(kk * div, cert_aim * 1000, cert_dev)) * exp_s
        # beyond the tick ends on the scale's +v side, which the mirror puts at -v
        vl = -(lens[4] / 2 + 45.0) if prof.tick_style == 'symmetric' else -(lens[4] + 45.0)
        oxl, oyl = (uk - Xc) * sig / mpp, (vl - Yc) / mpp
        pxl = cx + oxl * np.cos(th) - oyl * np.sin(th)
        pyl = cy + oxl * np.sin(th) + oyl * np.cos(th)
        label.paste(patch, (int(round(pxl - 160)), int(round(pyl - 70))), patch)
    chrome = np.maximum(chrome, np.asarray(label, float) / 255.0)
    chrome = ndi.gaussian_filter(chrome, cfg.psf_um / mpp)
    Tch = 1.0 - (1.0 - cfg.chrome_T) * chrome

    # --- grating
    rot = np.radians(stop['rot'])
    ang = cfg.alpha_rad + rot
    gx, gy = ug - cfg.grating_centre_um[0], vg - cfg.grating_centre_um[1]
    p = (np.cos(ang) * gx + np.sin(ang) * gy) / exp_g
    q = (-np.sin(ang) * gx + np.cos(ang) * gy) / exp_g
    fcell, fxc, fyc = cell
    if fyc is None:
        Nc = len(fcell)
        coords = [np.mod(p / cfg.period_x_um, 1.0) * Nc]
        fv = ndi.map_coordinates(fcell, coords, order=1, mode='grid-wrap')
        gv = np.abs(ndi.map_coordinates(fxc, coords, order=1, mode='grid-wrap'))
    else:
        Nc = fcell.shape[0]
        coords = [np.mod(q / cfg.period_y_um, 1.0) * Nc, np.mod(p / cfg.period_x_um, 1.0) * Nc]
        fv = ndi.map_coordinates(fcell, coords, order=1, mode='grid-wrap')
        gv = np.hypot(ndi.map_coordinates(fxc, coords, order=1, mode='grid-wrap'),
                      ndi.map_coordinates(fyc, coords, order=1, mode='grid-wrap'))
    b = np.clip(fv / np.maximum(gv, 1e-6) / mpp_g + 0.5, 0.0, 1.0)
    sd = cfg.defocus_um / mpp
    grad = ndi.gaussian_gradient_magnitude(b, sd) * sd / 0.3989
    lap = ndi.gaussian_laplace(b, sd) * sd * sd
    I_g = 1.0 - cfg.edge_loss * grad + cfg.edge_defocus * lap

    r2 = (xx - cx) ** 2 / (W / 2) ** 2 + (yy - cy) ** 2 / (H / 2) ** 2
    illum = (1.0 - 0.18 * r2) * (1.0 + 0.03 * (xx - cx) / W)
    img = cfg.signal * illum * Tch * I_g
    if cfg.noise:
        img = img + rng.normal(0, 1, img.shape) * np.sqrt(np.maximum(img, 0) + cfg.read_noise ** 2)
    return np.clip(np.rint(img), 0, 65535).astype(np.uint16)


def _render_job(args):
    cfg, prof, cell, aim, dev, stop, path = args
    import tifffile
    tifffile.imwrite(path, render(cfg, prof, cell, aim, dev, stop), compression='zlib')
    return path


# ---------------------------------------------------------------- whole run

CONFIG_TEMPLATE = """[paths]
run_csv = run.csv
images_dir = images
cert_csv = cert.csv
flat_field =
output_dir = output

[scale]
profile = {profile}
increases_to = {increases_to}
cert_units = mm
cert_temp_c = {cert_temp_c}
cte_ppm_per_c = {scale_cte}
cte_u_ppm_per_c = 1.0
cert_U_um_k2 = {cert_U_um}
cert_correlation = 0.0

[grating]
name = synthetic {grating}
type = {gtype}
period_x_um = {nominal_um}
period_y_um = {nominal_um}
orders_x = {orders_x}
orders_y = {orders_y}
cte_ppm_per_c = {grating_cte}
cte_u_ppm_per_c = 0.05
max_angle_deg = 2.0
centre_mm = {centre_mm}

[imaging]
um_per_px_nominal = 1.04
polarity = auto
optical_centre_px =
saturation_level =

[stage]
stage_to_scale_sign = 1

[analysis]
reference_temp_c = 20.0
temp_u_c = 0.3
outlier_sigma = 3.0
tick_fit_order = 3
tick_search_mm = 0.3
parallax_regressor = auto
ambiguity_threshold = 0.8
central_halfwidth_mm = 1.5
target_ppm = 50
harmonics_max = 10
workers = {workers}
"""


def make_run(out_dir, cfg, workers=1):
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    prof = scales.PROFILES[cfg.profile]
    rng = np.random.default_rng(cfg.seed)
    cell = make_cell(cfg)
    aim, dev = make_cert(cfg, prof)
    with open(os.path.join(out_dir, 'cert.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['aim_mm', 'measured_mm', 'U_um'])
        for a, d in zip(aim, dev):
            w.writerow([f"{a:.4f}", f"{a + d / 1000:.7f}", cfg.cert_U_um])
    gc_mm = cfg.grating_centre_um[0] / 1000
    rows, jobs = [], []
    for pi, (direction, rot) in enumerate(cfg.passes, start=1):
        L_along = cfg.period_x_um if rot % 180 == 0 else cfg.period_y_um
        step = cfg.step_periods * L_along
        xs = cfg.grating_centre_um[0] + (np.arange(cfg.n_stops) - (cfg.n_stops - 1) / 2) * step
        if direction == 'rev':
            xs = xs[::-1]
        T0 = cfg.temps_c[(pi - 1) % len(cfg.temps_c)]
        for si, x in enumerate(xs, start=1):
            Xc = x + rng.uniform(-cfg.stage_jitter_um, cfg.stage_jitter_um)
            Yc = cfg.baseline_offset_um + rng.uniform(-cfg.stage_jitter_um, cfg.stage_jitter_um)
            temp = T0 + 0.05 * (si - 1) / max(cfg.n_stops - 1, 1)
            stop = dict(Xc=Xc, Yc=Yc, rot=rot, temp_c=temp,
                        mpp_jitter=rng.normal(0, cfg.mpp_jitter_ppm * 1e-6),
                        seed=int(cfg.seed * 1000 + pi * 100 + si))
            name = f"pass{pi}_stop{si:02d}.tif"
            jobs.append((cfg, prof, cell, aim, dev, stop, os.path.join(out_dir, 'images', name)))
            rows.append(dict(file=name, **{'pass': pi}, stop=si, direction=direction,
                             grating_rot_deg=rot,
                             scale_mm=f"{Xc / 1000 + rng.normal(0, cfg.scale_mm_error):.3f}",
                             stage_x_mm=f"{Xc / 1000 + 37.2:.3f}", stage_y_mm='',
                             temp_c=f"{temp:.2f}", notes='synthetic'))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_render_job, jobs))
    else:
        for j in jobs:
            _render_job(j)
    with open(os.path.join(out_dir, 'run.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    orders = {'odd2d': ('odd', 'odd'), 'all2d': ('all', 'all'), 'mixed2d': ('odd', 'all'),
              'odd1d': ('odd', 'odd'), 'all1d': ('all', 'all')}[cfg.grating]
    with open(os.path.join(out_dir, 'config.ini'), 'w') as f:
        f.write(CONFIG_TEMPLATE.format(
            profile=cfg.profile, increases_to=cfg.increases_to, cert_temp_c=cfg.cert_temp_c,
            scale_cte=cfg.scale_cte, cert_U_um=cfg.cert_U_um, grating=cfg.grating,
            gtype='1d' if cfg.grating.endswith('1d') else '2d', nominal_um=cfg.nominal_um,
            orders_x=orders[0], orders_y=orders[1], grating_cte=cfg.grating_cte,
            centre_mm=f"{gc_mm:.4f}", workers=workers))
    truth = asdict(cfg)
    truth['note'] = "period_x_um / period_y_um are the true periods at 20 C"
    with open(os.path.join(out_dir, 'truth.json'), 'w') as f:
        json.dump(truth, f, indent=2)
    return truth


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic KeyMaster run.")
    ap.add_argument('out_dir')
    ap.add_argument('--profile', default='edmund59276_metric', choices=list(scales.PROFILES))
    ap.add_argument('--grating', default='odd2d',
                    choices=['odd2d', 'all2d', 'mixed2d', 'odd1d', 'all1d'])
    ap.add_argument('--passes', default='fwd:0,rev:0',
                    help="comma list of direction:rotation, e.g. fwd:0,rev:0,fwd:180,fwd:90")
    ap.add_argument('--stops', type=int, default=10)
    ap.add_argument('--increases-to', default='right', choices=['right', 'left'])
    ap.add_argument('--no-noise', action='store_true')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    passes = [(p.split(':')[0], int(p.split(':')[1])) for p in a.passes.split(',')]
    cfg = Synth(profile=a.profile, grating=a.grating, passes=passes, n_stops=a.stops,
                increases_to=a.increases_to, noise=not a.no_noise, seed=a.seed)
    if a.profile == 'sm4':
        cfg.grating_centre_um = (12500.0 - 4.2, 3.3)
    make_run(a.out_dir, cfg, a.workers)
    print(f"synthetic run written to {a.out_dir}")


if __name__ == '__main__':
    main()
