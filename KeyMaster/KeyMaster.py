# KeyMaster: grating period from Keyence microscope images against a certified stage micrometer.
#
# The scale lies chrome side down over the grating; every image shows scale ticks and grating
# pattern together. Lengths come only from the certified ticks; the microscope's um/px and stage
# readout only seed searches and tick numbering. For each image this script:
#   1. Reads the scale (ticks.py): baseline and angle, sub-pixel tick centres, absolute tick
#      numbers from the long-tick hierarchy, and a fit of certified position vs pixel.
#   2. Finds and locates the grating lattice (grating.py): FFT lattice search, a unit-cell
#      template from the first image of each pass, and a real-space fit of the template origin.
#   3. Converts that lattice point to scale coordinates.
# Across images (period.py) it counts periods stop to stop, fits the period with all passes
# pooled, corrects angle and temperature to 20 C, builds the uncertainty budget and checks
# forward/reverse and 0/180 deg consistency. Grating x and y axes are measured in separate
# runs (grating_rot_deg 0/180 vs 90/270) and reported separately.
#
# Quick mode (--quick) measures from one scale image and one grating image without stepping
# (about 100-150 ppm at k=2).
#
# Paths and options come from config.ini next to this script; see python KeyMaster.py --help.

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import configparser
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import scales
import ticks
import grating
import period

VERSION = "0.2.0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- configuration and inputs

def load_configuration(path):
    if not os.path.exists(path):
        sys.exit(f"[!] Configuration file not found: {path}")
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    cfg.read(path, encoding='utf-8-sig')    # tolerate a byte-order mark (Notepad, PowerShell)
    return cfg


def _f(section, key, default=None):
    v = section.get(key, fallback='').strip()
    return float(v) if v else default


def _resolve(path, base):
    if path and path.strip().lower() == 'none':
        return ''
    return path if (not path or os.path.isabs(path)) else os.path.join(base, path)


def settings(cfg):
    sc, gr, im, an, stg = cfg['scale'], cfg['grating'], cfg['imaging'], cfg['analysis'], \
        cfg['stage'] if cfg.has_section('stage') else {}
    oc = im.get('optical_centre_px', fallback='').strip()
    return dict(
        increases_to=sc.get('increases_to', 'right').strip().lower(),
        scale_cte=_f(sc, 'cte_ppm_per_c', 9.0), scale_cte_u=_f(sc, 'cte_u_ppm_per_c', 1.0),
        scale_T_cert=_f(sc, 'cert_temp_c', 20.0),
        dims=1 if gr.get('type', '2d').strip().lower() == '1d' else 2,
        period_x=_f(gr, 'period_x_um'), period_y=_f(gr, 'period_y_um') or _f(gr, 'period_x_um'),
        orders_x=gr.get('orders_x', 'auto').strip().lower(),
        orders_y=gr.get('orders_y', 'auto').strip().lower(),
        grating_cte=_f(gr, 'cte_ppm_per_c', 0.55), grating_cte_u=_f(gr, 'cte_u_ppm_per_c', 0.05),
        max_angle=np.radians(_f(gr, 'max_angle_deg', 2.0)),
        period_tol=_f(gr, 'period_tolerance_pct', 3.0) / 100,
        centre_mm=_f(gr, 'centre_mm'),
        mpp_nom=_f(im, 'um_per_px_nominal', 1.0),
        polarity=im.get('polarity', 'auto').strip().lower(),
        optical_centre=[float(v) for v in oc.split(',')] if oc else None,
        saturation=_f(im, 'saturation_level'),
        stage_sign=float(stg.get('stage_to_scale_sign', '1') or 1) if stg else 1.0,
        T_ref=_f(an, 'reference_temp_c', 20.0), temp_u=_f(an, 'temp_u_c', 0.3),
        outlier_sigma=_f(an, 'outlier_sigma', 3.0),
        fit_order=int(_f(an, 'tick_fit_order', 2)),
        search_mm=_f(an, 'tick_search_mm', 0.3),
        parallax=an.get('parallax_regressor', 'auto').strip().lower(),
        amb_threshold=_f(an, 'ambiguity_threshold', 0.8),
        central_hw_mm=_f(an, 'central_halfwidth_mm', 1.5),
        target_ppm=_f(an, 'target_ppm', 50.0),
        hmax=int(_f(an, 'harmonics_max', 10)),
        workers=int(_f(an, 'workers', 1)),
    )


def read_run(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or '').strip() for k, v in r.items() if k}
            if not r.get('file'):
                continue
            rows.append(r)
    if not rows:
        sys.exit(f"[!] No rows in {path}")
    # approximate scale reading: scale_mm, else extrapolated from stage_x_mm within the pass
    return rows


def fill_scale_mm(rows, stage_sign):
    by_pass = {}
    for r in rows:
        by_pass.setdefault(r.get('pass', '1'), []).append(r)
    for p, rs in by_pass.items():
        ref = next((r for r in rs if r.get('scale_mm')), None)
        for r in rs:
            if r.get('scale_mm'):
                r['_scale_mm'] = float(r['scale_mm'])
            elif ref is not None and r.get('stage_x_mm') and ref.get('stage_x_mm'):
                r['_scale_mm'] = float(ref['scale_mm']) + stage_sign * (
                    float(r['stage_x_mm']) - float(ref['stage_x_mm']))
            else:
                sys.exit(f"[!] {r['file']}: needs scale_mm, or stage_x_mm plus one scale_mm "
                         f"in pass {p}")


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        import tifffile
        img = tifffile.imread(path)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path))
    full = {np.dtype('uint8'): 255, np.dtype('uint16'): 65535}.get(img.dtype)
    img = img.astype(np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2) if img.shape[-1] in (3, 4) else img[0]
    return img, full


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- one stop

def process_stop(row, ctx, tmpl):
    """Measure one image. Returns (record, template); record['ok'] False on failure."""
    s = ctx['s']
    rot = int(float(row.get('grating_rot_deg') or 0)) % 360
    axis = 'x' if rot % 180 == 0 else 'y'
    rec = dict(file=row['file'], pass_=row.get('pass', '1'), stop=row.get('stop', ''),
               direction=row.get('direction', ''), rot=rot, axis=axis,
               temp_c=float(row['temp_c']) if row.get('temp_c') else np.nan,
               scale_mm_approx=row['_scale_mm'], ok=False, flags=[])
    rec['pass'] = rec.pop('pass_')
    try:
        img, full = load_image(os.path.join(ctx['images_dir'], row['file']))
        sat_level = s['saturation'] or (0.98 * full if full else np.inf)
        sat = img >= sat_level
        rec['saturated_frac'] = float(sat.mean())
        if rec['saturated_frac'] > 1e-3:
            rec['flags'].append(f"saturated pixels {100 * rec['saturated_frac']:.2f}%")
        norm, ink, pol = ticks.normalise(img, s['polarity'], ctx.get('flat'))
        tr = ticks.read_scale(ink, pol, ctx['profile'], ctx['cert'], s['mpp_nom'],
                              rec['scale_mm_approx'], s['increases_to'], s['fit_order'],
                              s['search_mm'])
        rec['flags'] += tr.flags
        mpp = tr.mpp
        p_along = s['period_x'] if axis == 'x' else s['period_y']
        p_perp = s['period_y'] if axis == 'x' else s['period_x']
        g, w = grating.prepare(norm, ticks.chrome_mask(ink), sat)
        r0 = np.array([tr.xc, tr.y0])
        gr, tmpl = grating.measure(g, w, r0, s['dims'], p_along / mpp, p_perp / mpp, tr.theta,
                                   s['max_angle'], tmpl, s['hmax'], s['period_tol'])
        rec['flags'] += gr.flags
        rec['repeat_reading'] = gr.discovery['reading']
        rec['period_mismatch_pct'] = gr.discovery['mismatch_pct']
        # Re-read the scale with the fitted grating pattern divided out of the tick band: the
        # texture under each tick otherwise biases its centroid by a few nm.
        half = 320.0 / s['mpp_nom']
        yb = tr.y0 + np.tan(tr.theta) * (np.array([0, img.shape[1] - 1]) - tr.xc)
        lo = int(max(0, np.floor(yb.min() - half)))
        hi = int(min(img.shape[0], np.ceil(yb.max() + half)))
        model = grating.model_rows(img.shape, lo, hi, gr.G, gr.p, r0, tmpl, gr.amp, gr.offset)
        norm2 = norm.copy()
        norm2[lo:hi] = (1 + norm[lo:hi]) / np.maximum(1 + model, 0.2) - 1
        ink2 = -norm2 if pol == 'dark' else norm2
        tr = ticks.read_scale(ink2, pol, ctx['profile'], ctx['cert'], s['mpp_nom'],
                              rec['scale_mm_approx'], s['increases_to'], s['fit_order'],
                              s['search_mm'])
        mpp = tr.mpp
        e, ep = tr.e, tr.e_perp
        if s['dims'] == 2:
            q = r0 + gr.p
            s_q, t_q = (float(v) for v in tr.st(q[0], q[1]))
            a1, a2 = gr.A
            B = tr.dXds(s_q)
            A1 = np.array([B * a1 @ e, mpp * a1 @ ep])
            A2 = np.array([B * a2 @ e, mpp * a2 @ ep])
            alpha = float(np.arctan2(a1 @ ep, a1 @ e))
            var_s = float(e @ gr.cov_p @ e)
            period_px = float(np.hypot(*a1))
            rec['axes_angle_deg'] = float(np.degrees(np.arccos(
                a1 @ a2 / (np.hypot(*a1) * np.hypot(*a2)))))
        else:
            g1 = gr.G[0]
            ge = float(g1 @ e)
            delta = float(gr.p[0])
            s_q, t_q = (delta - np.rint(delta)) / ge, 0.0
            q = np.array(tr.xy(s_q, 0.0))
            B = tr.dXds(s_q)
            A1, A2 = np.array([B / ge, 0.0]), np.zeros(2)
            alpha = float(np.arctan2(g1 @ ep, g1 @ e))
            var_s = float(gr.cov_p[0, 0]) / ge ** 2
            period_px = float(1 / np.hypot(*g1))
        oc = s['optical_centre'] or [(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2]
        s_opt = float(tr.st(oc[0], oc[1])[0])
        rec.update(
            ok=True, polarity=pol, theta_deg=float(np.degrees(tr.theta)), mpp_um=float(mpp),
            n_ticks=int(tr.used.sum()), tick_rms_um=tr.rms_um,
            k_first=int(tr.k.min()), k_last=int(tr.k.max()),
            assign_score=tr.assign_score, assign_margin=tr.assign_margin,
            scale_quad_ppm_per_mm=float(2 * tr.coef[-3] / mpp * 1e3 / abs(tr.coef[-2]) * 1e6)
            if s['fit_order'] >= 2 else 0.0,
            point_x_px=float(q[0]), point_y_px=float(q[1]), s_px=s_q, t_px=t_q,
            X=float(tr.X_at(s_q)), X_lin=float(tr.X_at(s_q, linear=True)), Y=t_q * mpp,
            # offset from the optical axis in scale coordinates (sign follows the scale)
            d_um=float(tr.X_at(s_q) - tr.X_at(s_opt)),
            sigma_X_um=float(abs(B) * np.sqrt(max(var_s, 0))),
            A1=A1, A2=A2, alpha=alpha, period_px=period_px,
            period_local_um=period_px * mpp, ratios=gr.ratios, corr=gr.corr,
            fit_rms=gr.fine_rms, n_harm=gr.n_harm, parity=dict(tmpl.parity),
            _ticks=dict(s=tr.s, resid=tr.resid_um, used=tr.used, theta=tr.theta, xc=tr.xc,
                        y0=tr.y0, shape=img.shape, A=gr.A, q=q))
        if tr.rms_um > 0.1:
            rec['flags'].append(f"tick-fit residual {tr.rms_um:.3f} um > 0.1 um")
    except Exception as ex:   # keep going; the stop is reported as failed
        rec['flags'].append(f"FAILED: {ex}")
    return rec, tmpl


def _worker(args):
    row, ctx, tmpl = args
    rec, _ = process_stop(row, ctx, tmpl)
    return rec


def process_all(rows, ctx, workers):
    by_pass = {}
    for r in rows:
        by_pass.setdefault(r.get('pass', '1'), []).append(r)
    records = []
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for p, rs in by_pass.items():
            tmpl = None
            rest = list(rs)
            while rest and tmpl is None:   # first usable image of the pass builds the template
                r = rest.pop(0)
                rec, t = process_stop(r, ctx, None)
                records.append(rec)
                print(f"  pass {p} {r['file']}: " + ("template" if rec['ok'] else rec['flags'][-1]))
                if rec['ok']:
                    tmpl = t
            if not rest:
                continue
            if pool:
                recs = list(pool.map(_worker, [(r, ctx, tmpl) for r in rest]))
            else:
                recs = [process_stop(r, ctx, tmpl)[0] for r in rest]
            for r, rec in zip(rest, recs):
                print(f"  pass {p} {r['file']}: " + (f"X = {rec['X'] / 1000:.4f} mm"
                                                     if rec['ok'] else rec['flags'][-1]))
            records += recs
    finally:
        if pool:
            pool.shutdown()
    return records


# ---------------------------------------------------------------- per-axis analysis

def analyse_axis(stops, s, cert, axis):
    out = dict(axis=axis, n_stops=len(stops))
    eff, rmed = period.effective_lattice([st['ratios'] for st in stops], s['dims'],
                                         s['amb_threshold'])
    out['effective_lattice'] = eff
    out['half_shift_ratios'] = rmed
    for p in sorted({st['pass'] for st in stops}):
        period.count_pass([st for st in stops if st['pass'] == p], s['dims'], eff)
    # Temperature, per stop: scale readings to true lengths at the image's temperature (scale
    # expands about its zero), then the grating lattice back to 20 C about the grating centre.
    temps_all = np.array([st['temp_c'] for st in stops])
    T_all = float(np.nanmean(temps_all)) if np.any(np.isfinite(temps_all)) else s['T_ref']
    xs = [st['X'] for st in stops]
    Xg0 = s['centre_mm'] * 1000 if s['centre_mm'] is not None else (max(xs) + min(xs)) / 2

    def to20(st, X):
        T = st['temp_c'] if np.isfinite(st['temp_c']) else T_all
        Xp = X * (1 + s['scale_cte'] * 1e-6 * (T - s['scale_T_cert']))
        fg = s['grating_cte'] * 1e-6 * (T - s['T_ref'])
        return Xp - (Xp - Xg0) * fg / (1 + fg)

    for st in stops:
        st['Xt_raw'] = to20(st, st['X'])
    # Gap parallax self-calibration: the period measured inside each image (scale-plane um/px)
    # is Lambda / (1 + eps); the stepped fit gives Lambda. Correct each point by eps * d.
    alpha_all = float(np.mean([st['alpha'] for st in stops]))
    fit0 = period.fit_period([dict(st, flags=[]) for st in stops], False, s['outlier_sigma'],
                             x_key='Xt_raw')
    if fit0 is None:
        out['error'] = "not enough stops spanning at least two periods"
        return out
    local = np.median([st['period_local_um'] for st in stops]) * \
        period.temperature_correct(1.0, T_all, s['scale_cte'], s['scale_T_cert'],
                                   s['grating_cte'], s['T_ref'])
    eps = period.to_period(fit0['S'], alpha_all, s['dims']) / local - 1
    for st in stops:
        st['Xc'] = st['X'] + eps * st['d_um']
        st['Xt'] = to20(st, st['Xc'])
        st['Xt_lin'] = to20(st, st['X_lin'] + eps * st['d_um'])
    out['gap_magnification_ppm'] = eps * 1e6
    d_spread = np.std([st['d_um'] for st in stops])
    use_par = s['parallax'] == 'on' or (s['parallax'] == 'auto' and d_spread > 40 and
                                        len(stops) >= 8)
    fit = period.fit_period(stops, use_par, s['outlier_sigma'], x_key='Xt')
    if fit is None:
        out['error'] = "not enough stops spanning at least two periods"
        return out
    for st, u in zip(stops, fit['used']):
        st['used'] = bool(u)
    for st, r in zip(stops, fit['residuals']):
        st['residual_um'] = float(r)
    used = [st for st in stops if st['used']]
    alphas = np.array([st['alpha'] for st in used])
    alpha_mean = float(np.mean(alphas))
    alpha_u = float(np.hypot(np.std(alphas, ddof=1) / np.sqrt(len(alphas)), 1e-4))
    temps = np.array([st['temp_c'] for st in used])
    T_run = float(np.nanmean(temps)) if np.any(np.isfinite(temps)) else s['T_ref']
    cfg_t = dict(scale_cte=s['scale_cte'], scale_cte_u=s['scale_cte_u'],
                 scale_T_cert=s['scale_T_cert'], grating_cte=s['grating_cte'],
                 grating_cte_u=s['grating_cte_u'], T_ref=s['T_ref'], temp_u=s['temp_u'])

    def L20_of(S):    # positions are already at 20 C; only the angle remains
        return period.to_period(S, alpha_mean, s['dims'])

    L20 = L20_of(fit['S'])
    L_axis = abs(fit['S'])
    L_T = L20 * (1 + s['grating_cte'] * 1e-6 * (T_run - s['T_ref']))
    # model sensitivities: linear scale fit; no gap correction and no regressor
    stops_copy = [dict(st, flags=[]) for st in used]
    alt = period.fit_period(stops_copy, use_par, 1e9, x_key='Xt_lin')
    L_alt_tick = L20_of(alt['S']) if alt else L20
    alt2 = period.fit_period([dict(st, flags=[]) for st in used], False, 1e9, x_key='Xt_raw')
    L_alt_par = L20_of(alt2['S']) if alt2 else None
    slopes = period.per_pass_slopes(stops, fit['beta'], key='Xt')
    x_used = [st['X'] for st in used]
    span = max(x_used) - min(x_used)
    terms, uc = period.uncertainty_budget(L20, fit, slopes, used, cert, cfg_t, s['dims'],
                                          alpha_mean, alpha_u, L_alt_tick, L_alt_par, T_run, span)
    U = 2 * uc
    out.update(
        period_20C_um=L20, U_k2_um=U, U_k2_ppm=U / L20 * 1e6, u_k1_um=uc,
        target_ppm=s['target_ppm'], pass_fail='PASS' if U / L20 * 1e6 <= s['target_ppm'] else 'FAIL',
        period_along_axis_um=L_axis, period_at_T_um=L_T, T_run_c=T_run,
        alpha_mrad=alpha_mean * 1e3, alpha_u_mrad=alpha_u * 1e3,
        nominal_period_um=s['period_x'] if axis == 'x' else s['period_y'],
        span_mm=span / 1000, n_used=len(used), fit_sigma_um=fit['sigma'], dof=fit['dof'],
        parallax_regressor=use_par, parallax_beta_ppm=fit['beta'] * 1e6,
        parallax_beta_u_ppm=fit['se_beta'] * 1e6,
        budget=[dict(term=n, u_k1_um=u, u_k1_ppm=u / L20 * 1e6, U_k2_ppm=2 * u / L20 * 1e6,
                     note=note) for n, u, note in terms],
        per_pass={p: dict(v, period_um=period.to_period(v['S'], alpha_mean, s['dims']))
                  for p, v in slopes.items()},
        consistency=[c for c in (period.compare_groups(slopes, 'direction', 'fwd', 'rev'),
                                 period.compare_groups(slopes, 'rot', 0 if axis == 'x' else 90,
                                                       180 if axis == 'x' else 270)) if c],
        harmonic_energy_by_parity={k: float(np.mean([st['parity'][k] for st in used]))
                                   for k in used[0]['parity']},
        fit=dict(S=fit['S'], se_S=fit['se_S'], intercepts=fit['intercepts']),
    )
    out['deviation_from_nominal_ppm'] = (L20 / out['nominal_period_um'] - 1) * 1e6
    # orders check
    orders = s['orders_x'] if axis == 'x' else s['orders_y']
    if s['dims'] == 2 and orders in ('odd', 'all'):
        sym = eff in ('centred', 'half_both')
        if orders == 'all' and sym:
            out.setdefault('warnings', []).append(
                "config says all orders, but the images repeat at half a period")
    # central uniformity check
    c_mm = s['centre_mm'] if s['centre_mm'] is not None else (max(x_used) + min(x_used)) / 2000
    central = [dict(st, flags=[]) for st in used
               if abs(st['X'] / 1000 - c_mm) <= s['central_hw_mm']]
    cfit = period.fit_period(central, False, 1e9, x_key='Xt') if len(central) >= 3 else None
    if cfit:
        Lc = L20_of(cfit['S'])
        out['central'] = dict(centre_mm=c_mm, halfwidth_mm=s['central_hw_mm'], n=len(central),
                              period_20C_um=Lc, u_stat_um=cfit['se_S'] * Lc / abs(cfit['S']),
                              difference_ppm=(Lc / L20 - 1) * 1e6)
    else:
        out['central'] = dict(centre_mm=c_mm, halfwidth_mm=s['central_hw_mm'], n=len(central),
                              note="too few stops in the central zone for a fit")
    return out


# ---------------------------------------------------------------- outputs

STOP_COLUMNS = ['file', 'pass', 'stop', 'direction', 'rot', 'axis', 'ok', 'used', 'temp_c',
                'scale_mm_approx', 'mpp_um', 'theta_deg', 'n_ticks', 'tick_rms_um', 'k_first',
                'k_last', 'assign_score', 'assign_margin', 'scale_quad_ppm_per_mm',
                'point_x_px', 'point_y_px', 'X', 'X_lin', 'Xt', 'Y', 'd_um', 'sigma_X_um', 'n1',
                'n2', 'count_frac', 'residual_um', 'alpha', 'period_px', 'period_local_um',
                'fit_rms', 'repeat_reading', 'period_mismatch_pct', 'saturated_frac', 'flags']


def write_stops(path, records):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        names = {'X': 'X_um', 'X_lin': 'X_linfit_um', 'Xt': 'X_corrected_20C_um', 'Y': 'Y_um',
                 'alpha': 'alpha_mrad'}
        w.writerow([names.get(c, c) for c in STOP_COLUMNS] + ['ratio_10', 'ratio_01', 'ratio_11'])
        for r in records:
            row = []
            for c in STOP_COLUMNS:
                v = r.get(c, '')
                if c == 'alpha' and v != '':
                    v = v * 1e3
                if c == 'flags':
                    v = '; '.join(v)
                if isinstance(v, float):
                    v = f"{v:.6g}" if c not in ('X', 'X_lin', 'Xt') else f"{v:.4f}"
                row.append(v)
            rat = r.get('ratios') or {}
            row += [f"{rat[k]:.3f}" if k in rat else '' for k in ('10', '01', '11')]
            w.writerow(row)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def summary_text(res):
    L = []
    L.append(f"KeyMaster {VERSION}  {res['timestamp']}")
    L.append(f"Grating {res['grating']}  |  scale {res['scale']['model']}  |  "
             f"certificate: {res['scale']['certificate']}")
    if not res['scale']['certified']:
        L.append("!! No certificate loaded: nominal tick positions, result is NOT traceable.")
    if res['scale']['layout_assumed']:
        L.append("!! Scale tick layout is assumed (not yet checked on the real part).")
    L.append(f"Images: {res['n_images']} ({res['n_failed']} failed)  run time "
             f"{res['run_time_s']:.0f} s")
    for ax in res['axes']:
        L.append("")
        L.append(f"Grating {ax['axis']} axis:")
        if 'error' in ax:
            L.append(f"  {ax['error']}")
            continue
        L.append(f"  Period at {res['reference_temp_c']:.0f} C: {ax['period_20C_um']:.5f} um "
                 f"+/- {ax['U_k2_um']:.5f} um (k=2) = {ax['U_k2_ppm']:.1f} ppm  "
                 f"[{ax['pass_fail']} vs {ax['target_ppm']:.0f} ppm]")
        L.append(f"  Nominal {ax['nominal_period_um']} um: deviation "
                 f"{ax['deviation_from_nominal_ppm']:+.0f} ppm")
        L.append(f"  {ax['n_used']}/{ax['n_stops']} stops, span {ax['span_mm']:.2f} mm, "
                 f"fit sigma {ax['fit_sigma_um'] * 1e3:.1f} nm, alpha {ax['alpha_mrad']:.2f} "
                 f"mrad, T {ax['T_run_c']:.2f} C, lattice used for counting: "
                 f"{ax['effective_lattice']}, grating-plane magnification "
                 f"{ax['gap_magnification_ppm']:+.0f} ppm")
        L.append("  Budget (U k=2, ppm): " + "; ".join(
            f"{b['term']} {b['U_k2_ppm']:.1f}" for b in ax['budget']))
        for c in ax['consistency']:
            L.append(f"  {c['groups']}: difference {c['difference_um'] * 1e3:+.1f} nm/period, "
                     f"z = {c['z']:+.2f} -> {'consistent' if c['consistent'] else 'INCONSISTENT'}")
        cen = ax['central']
        if 'period_20C_um' in cen:
            L.append(f"  Central +/-{cen['halfwidth_mm']} mm: {cen['period_20C_um']:.5f} um "
                     f"({cen['difference_ppm']:+.0f} ppm, u_stat "
                     f"{cen['u_stat_um'] / cen['period_20C_um'] * 1e6:.0f} ppm, n={cen['n']})")
        else:
            L.append(f"  Central +/-{cen['halfwidth_mm']} mm: {cen['note']} (n={cen['n']})")
        for wmsg in ax.get('warnings', []):
            L.append(f"  ! {wmsg}")
    flagged = [r for r in res['stops_flagged']]
    if flagged:
        L.append("")
        L.append("Flags:")
        for f in flagged:
            L.append(f"  {f}")
    return "\n".join(L)


# ---------------------------------------------------------------- main run

def run(args):
    cfg_path = args.config or os.path.join(SCRIPT_DIR, 'config.ini')
    cfg = load_configuration(cfg_path)
    base = os.path.dirname(os.path.abspath(cfg_path))
    paths = cfg['paths']
    s = settings(cfg)
    if args.workers:
        s['workers'] = args.workers
    run_csv = _resolve(args.run or paths.get('run_csv', 'run.csv'), base)
    images_dir = _resolve(args.images or paths.get('images_dir', '.'), base)
    cert_csv = _resolve(args.cert if args.cert is not None else paths.get('cert_csv', ''), base)
    out_root = _resolve(args.out or paths.get('output_dir', 'output'), base)
    flat_path = _resolve(paths.get('flat_field', '').strip(), base)

    profile = scales.load_profile(cfg['scale'])
    sc = cfg['scale']
    U_def = _f(sc, 'cert_U_um_k2', 0.415)
    rho = _f(sc, 'cert_correlation', 0.0)
    if cert_csv:
        cert = scales.Certificate.from_csv(cert_csv, sc.get('cert_units', 'mm').strip(),
                                           U_def, rho)
    else:
        cert = scales.Certificate.nominal(U_def, rho)
    rows = read_run(run_csv)
    fill_scale_mm(rows, s['stage_sign'])
    flat = load_image(flat_path)[0] if flat_path else None

    t0 = time.time()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, stamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"KeyMaster {VERSION}: {len(rows)} images, scale {profile.model}, "
          f"{'1D' if s['dims'] == 1 else '2D'} grating")
    ctx = dict(s=s, profile=profile, cert=cert, images_dir=images_dir, flat=flat)
    records = process_all(rows, ctx, s['workers'])

    axes = []
    for axis in ('x', 'y'):
        st = [r for r in records if r['ok'] and r['axis'] == axis]
        if st:
            axes.append(analyse_axis(st, s, cert, axis))
    hashes = {os.path.basename(p): file_hash(p) for p in
              [run_csv, cfg_path] + ([cert_csv] if cert_csv else []) +
              [os.path.join(images_dir, r['file']) for r in rows
               if os.path.exists(os.path.join(images_dir, r['file']))]}
    res = dict(
        version=VERSION, timestamp=datetime.now().isoformat(timespec='seconds'),
        grating=cfg['grating'].get('name', ''), reference_temp_c=s['T_ref'],
        scale=dict(model=profile.model, profile=profile.name, division_um=profile.division_um,
                   tick_levels=profile.tick_levels, layout_assumed=profile.layout_assumed,
                   increases_to=s['increases_to'], certificate=cert.source,
                   certified=cert.certified),
        n_images=len(records), n_failed=sum(not r['ok'] for r in records),
        run_time_s=time.time() - t0, axes=axes,
        stops_flagged=[f"{r['file']}: {'; '.join(r['flags'])}" for r in records if r['flags']],
        input_sha256=hashes, config=dict({k: dict(cfg[k]) for k in cfg.sections()}))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(jsonable(res), f, indent=2)
    write_stops(os.path.join(out_dir, 'stops.csv'), records)
    per_axis = {a['axis']: dict(period_um=a['period_20C_um'], u_ppm=a['u_k1_um'] / a['period_20C_um'] * 1e6,
                                terms={b['term']: b['u_k1_ppm'] for b in a['budget']})
                for a in axes if 'error' not in a}
    angles = [r['axes_angle_deg'] for r in records if r['ok'] and 'axes_angle_deg' in r]
    write_period_file(out_dir, cfg, s, per_axis, angles, method="stepped, certified stage micrometer",
                      traceable=cert.certified,
                      length_reference=f"{profile.model}; certificate {cert.source}",
                      notes=(["scale tick layout assumed, not yet checked on the part"]
                             if profile.layout_assumed else []))
    text = summary_text(res)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    try:
        import plots
        plots.make_all(out_dir, records, axes, images_dir, s)
    except Exception as ex:
        print(f"[!] plotting failed: {ex}")
    print()
    print(text)
    print(f"\nOutputs: {out_dir}")
    return res, out_dir


# ---------------------------------------------------------------- period file for ArgusEye

# budget terms common to the x and y runs (same scale, same thermal model)
SHARED_TERMS = ('scale certificate', 'certificate interpolation', 'temperature',
                'Keyence calibration', 'lens distortion')


def write_period_file(out_dir, cfg, s, per_axis, axes_angles, method, traceable,
                      length_reference, notes=()):
    """grating_period.json: the measured period per grating axis, in ArgusEye's format.

    per_axis: {'x': dict(period_um, u_ppm (k=1), terms {name: u_ppm})}, 'y' optional."""
    gr = cfg['grating']

    def ax(a):
        v = per_axis.get(a)
        if not v:
            return None, None, None
        return float(v['period_um']), float(v['u_ppm']), float(2 * v['u_ppm'])

    px, upx, Upx = ax('x')
    py, upy, Upy = ax('y')
    rho = None
    if px is not None and py is not None:
        tx, ty = per_axis['x']['terms'], per_axis['y']['terms']
        shared = sum(tx[k] * ty[k] for k in tx if k in ty and k.startswith(SHARED_TERMS))
        rho = float(np.clip(shared / (upx * upy), -1, 1))
    if axes_angles:
        ang = float(np.mean(axes_angles))
        u_ang = float(np.hypot(np.std(axes_angles, ddof=1) / np.sqrt(len(axes_angles))
                               if len(axes_angles) > 1 else 0.0, 0.02))
    else:
        ang, u_ang = 90.0, None
    data = {
        "grating": gr.get('name', '').strip(),
        "period_x_um": px, "u_period_x_ppm": upx, "U_period_x_ppm": Upx,
        "period_y_um": py, "u_period_y_ppm": upy, "U_period_y_ppm": Upy,
        "y_measured": py is not None,
        "correlation_xy": rho,
        "axes_angle_deg": ang, "u_axes_angle_deg": u_ang,
        "axis_definition": gr.get('axis_definition', '').strip(),
        "reference_temp_c": float(s['T_ref']),
        "cte_ppm_per_c": float(s['grating_cte']),
        "traceable": bool(traceable),
        "length_reference": length_reference,
        "method": method,
        "measured": {"date": datetime.now().isoformat(timespec='seconds'),
                     "run_folder": os.path.abspath(out_dir), "keymaster_version": VERSION},
        "notes": list(notes),
    }
    path = os.path.join(out_dir, 'grating_period.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path, data


# ---------------------------------------------------------------- Keyence-calibration mode

def keyence_mode(args):
    """Period from the Keyence's own calibration (no micrometer). NOT traceable.

    Every image is analysed on its own: the lattice period in pixels times the um/px that the
    VHX stored in the TIFF (or [keyence] um_per_px). The images should be at the same lens
    and zoom and taken after the latest calibration."""
    import glob
    import keyence
    cfg_path = args.config or os.path.join(SCRIPT_DIR, 'config.ini')
    cfg = load_configuration(cfg_path)
    base = os.path.dirname(os.path.abspath(cfg_path))
    s = settings(cfg)
    kc = cfg['keyence'] if cfg.has_section('keyence') else {}
    cal_u = float(kc.get('calibration_u_ppm', '5000') or 5000)
    dist_u = float(kc.get('distortion_u_ppm', '100') or 100)
    factor = float(kc.get('tag17_factor', '') or keyence.TAG17_FACTOR)
    mpp_override = float(kc['um_per_px']) if kc.get('um_per_px', '').strip() else None
    files = []
    for pat in args.keyence:
        hits = sorted(glob.glob(pat))
        files += hits if hits else [pat]
    rot = args.rot % 360
    axis = 'x' if rot % 180 == 0 else 'y'
    p_al = s['period_x'] if axis == 'x' else s['period_y']
    p_pe = s['period_y'] if axis == 'x' else s['period_x']
    T = args.temp
    out_root = _resolve(args.out or cfg['paths'].get('output_dir', 'output'), base)
    out_dir = os.path.join(out_root, datetime.now().strftime('%Y%m%d_%H%M%S') + '_keyence')
    os.makedirs(out_dir, exist_ok=True)
    print(f"KeyMaster {VERSION} KEYENCE-CALIBRATION mode (not traceable): {len(files)} images, "
          f"grating {axis} axis")
    recs = []
    for f in files:
        rec = dict(file=os.path.basename(f), ok=False, flags=[])
        try:
            meta = keyence.read_meta(f, factor)
            if mpp_override:
                mpp = mpp_override
                rec['flags'].append("um/px from config, not from the file")
            elif meta:
                mpp = meta['um_per_px']
            else:
                raise RuntimeError("no Keyence calibration in the file; set [keyence] um_per_px")
            rec.update(meta or {})
            img, full = load_image(f)
            sat = img >= (s['saturation'] or (0.98 * full if full else np.inf))
            rec['saturated_frac'] = float(sat.mean())
            norm, ink, pol = ticks.normalise(img, 'dark' if s['polarity'] == 'auto'
                                             else s['polarity'])
            from scipy import ndimage as ndi
            dirt = ndi.binary_dilation(ndi.binary_opening(ink > 0.6, iterations=2), iterations=6)
            g, w = grating.prepare(norm, dirt, sat)
            H, W = img.shape
            r0 = np.array([(W - 1) / 2, (H - 1) / 2])
            gr, _ = grating.measure(g, w, r0, s['dims'], p_al / mpp, p_pe / mpp, 0.0,
                                    s['max_angle'], None, s['hmax'], s['period_tol'])
            rec['flags'] += gr.flags
            if s['dims'] == 2:
                a1, a2 = gr.A
                L_px, L2_px = float(np.hypot(*a1)), float(np.hypot(*a2))
                rec['axes_angle_deg'] = float(np.degrees(np.arccos(a1 @ a2 / (L_px * L2_px))))
                rec['alpha_deg'] = float(np.degrees(np.arctan2(a1[1], a1[0])))
                rec['period_other_axis_um'] = L2_px * mpp
            else:
                L_px = float(1 / np.hypot(*gr.G[0]))
                rec['alpha_deg'] = float(np.degrees(np.arctan2(gr.G[0][1], gr.G[0][0])))
            LT = L_px * mpp
            rec.update(ok=True, um_per_px_used=mpp, period_px=L_px, period_T_um=LT,
                       period_20C_um=LT / (1 + s['grating_cte'] * 1e-6 *
                                           ((T if T is not None else s['T_ref']) - s['T_ref'])),
                       repeat_reading=gr.discovery['reading'],
                       period_mismatch_pct=gr.discovery['mismatch_pct'], fit_rms=gr.fine_rms)
            print(f"  {rec['file']}: zoom {rec.get('zoom')}, {mpp:.6f} um/px, period "
                  f"{L_px:.3f} px = {rec['period_20C_um']:.4f} um ({gr.discovery['reading']})")
        except Exception as ex:
            rec['flags'].append(f"FAILED: {ex}")
            print(f"  {rec['file']}: {rec['flags'][-1]}")
        recs.append(rec)
    ok = [r for r in recs if r['ok']]
    if not ok:
        sys.exit("[!] no image could be measured")
    # combine: mean per zoom, then the mean of the zoom means
    zooms = {}
    for r in ok:
        zooms.setdefault(r.get('zoom'), []).append(r['period_20C_um'])
    zmeans = {z: float(np.mean(v)) for z, v in zooms.items()}
    L = float(np.mean(list(zmeans.values())))
    # image-to-image scatter within each zoom (zoom differences have their own term)
    dev = np.concatenate([np.array(v) - np.mean(v) for v in zooms.values()])
    dof = len(dev) - len(zooms)
    terms = {}
    terms['Keyence calibration'] = cal_u
    terms['statistical (image to image)'] = (float(np.sqrt(np.sum(dev ** 2) / dof)
                                                   / np.sqrt(len(dev)) / L * 1e6) if dof > 0 else 0.0)
    if dof <= 0:
        notes_stat = "only one image per zoom: image-to-image scatter not measured"
    else:
        notes_stat = None
    if len(zmeans) > 1:
        zr = max(zmeans.values()) - min(zmeans.values())
        terms['zoom-to-zoom disagreement'] = zr / 2 / L * 1e6 / np.sqrt(3)
    terms['lens distortion (in-image average)'] = dist_u
    terms['temperature'] = s['grating_cte'] * (s['temp_u'] if T is not None else 2.0)
    uc = float(np.sqrt(sum(v ** 2 for v in terms.values())))
    notes = ["Keyence calibration only; replace with a stepped micrometer measurement",
             f"um/px = Keyence entry 17 / {factor} (measured at x100 on 2026-11-09)"]
    if T is None:
        notes.append("temperature not given; 20 +/- 2 C assumed")
    if notes_stat:
        notes.append(notes_stat)
    widths = {(r.get('image_width_px'), r.get('calibration_width_px')) for r in ok
              if abs(r.get('width_scale', 1.0) - 1) > 1e-9}
    for w_img, w_cal in sorted(widths):
        notes.append(f"images saved {w_img} px wide: um/px scaled by {w_cal}/{w_img} (same field "
                     f"of view as the {w_cal} px image; checked to 45 ppm on 2026-11-13)")
    angles = [r['axes_angle_deg'] for r in ok if 'axes_angle_deg' in r]
    model = next((r.get('model') for r in ok if r.get('model')), 'VHX')
    path, data = write_period_file(
        out_dir, cfg, s, {axis: dict(period_um=L, u_ppm=uc, terms=terms)}, angles,
        method="Keyence calibration, single images (no micrometer)", traceable=False,
        length_reference=f"Keyence {model} calibration (OP-88592 scale)", notes=notes)
    res = dict(version=VERSION, mode='keyence', axis=axis, period_20C_um=L, u_k1_ppm=uc,
               U_k2_ppm=2 * uc, budget_k1_ppm=terms, per_zoom=zmeans, images=recs,
               input_sha256={os.path.basename(f): file_hash(f) for f in files
                             if os.path.exists(f)})
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(jsonable(res), f, indent=2)
    print(f"\n  Period ({axis} axis) at {s['T_ref']:.0f} C: {L:.4f} um +/- {2 * uc * L * 1e-6:.4f} um "
          f"(k=2) = {2 * uc:.0f} ppm   [Keyence calibration, NOT traceable]")
    for z, m in sorted(zmeans.items(), key=lambda v: (v[0] is None, v[0])):
        print(f"    zoom {z}: {m:.4f} um ({len(zooms[z])} images)")
    for k, v in terms.items():
        print(f"    {k}: {2 * v:.0f} ppm (k=2)")
    if angles:
        print(f"  Angle between the lattice axes: {data['axes_angle_deg']:.4f} deg")
    print(f"\n  Period file: {path}")
    return data


# ---------------------------------------------------------------- quick mode

def quick(args):
    cfg_path = args.config or os.path.join(SCRIPT_DIR, 'config.ini')
    cfg = load_configuration(cfg_path)
    base = os.path.dirname(os.path.abspath(cfg_path))
    s = settings(cfg)
    profile = scales.load_profile(cfg['scale'])
    sc = cfg['scale']
    cert_csv = _resolve(args.cert if args.cert is not None else cfg['paths'].get('cert_csv', ''),
                        base)
    U_def, rho = _f(sc, 'cert_U_um_k2', 0.415), _f(sc, 'cert_correlation', 0.0)
    cert = (scales.Certificate.from_csv(cert_csv, sc.get('cert_units', 'mm').strip(), U_def, rho)
            if cert_csv else scales.Certificate.nominal(U_def, rho))
    qs = cfg['quick'] if cfg.has_section('quick') else {}
    height_u = float(qs.get('height_u_ppm', '50') or 50)
    parallax_ppm = float(qs.get('parallax_ppm', '0') or 0)
    scale_path = args.quick[0]
    grat_path = args.quick[1] if len(args.quick) > 1 else args.quick[0]
    same = os.path.abspath(scale_path) == os.path.abspath(grat_path)
    img, full = load_image(scale_path)
    norm, ink, pol = ticks.normalise(img, s['polarity'])
    if args.scale_mm is None:
        sys.exit("[!] quick mode needs --scale-mm (approximate scale reading at the image centre)")
    tr = ticks.read_scale(ink, pol, profile, cert, s['mpp_nom'], args.scale_mm,
                          s['increases_to'], s['fit_order'], s['search_mm'])
    if same:
        gimg, gnorm, gink = img, norm, ink
        chrome = ticks.chrome_mask(ink)
    else:
        gimg, gfull = load_image(grat_path)
        gnorm, gink, _ = ticks.normalise(gimg, pol)
        chrome = np.zeros(gimg.shape, bool)
    sat = gimg >= (s['saturation'] or (0.98 * full if full else np.inf))
    g, w = grating.prepare(gnorm, chrome, sat)
    axis = 'x' if args.rot % 180 == 0 else 'y'
    p_al = s['period_x'] if axis == 'x' else s['period_y']
    p_pe = s['period_y'] if axis == 'x' else s['period_x']
    r0 = np.array([tr.xc, tr.y0])
    gr, _ = grating.measure(g, w, r0, s['dims'], p_al / tr.mpp, p_pe / tr.mpp, tr.theta,
                            s['max_angle'], None, s['hmax'], s['period_tol'])
    print(f"  image repeats every {', '.join(f'{v:.3f}' for v in gr.discovery['repeat_px'])} px: "
          f"read as {gr.discovery['reading']} ({gr.discovery['mismatch_pct']:+.2f}% from nominal)")
    # um/px averaged over the grating region along the axis
    ys, xs = np.nonzero(w)
    sv, _ = tr.st(xs, ys)
    lo, hi = np.percentile(sv, [10, 90])
    mpp_eff = abs(tr.X_at(hi) - tr.X_at(lo)) / (hi - lo)
    mpp_lin = abs(tr.X_at(hi, True) - tr.X_at(lo, True)) / (hi - lo)
    if s['dims'] == 2:
        a1 = gr.A[0]
        L_px = float(np.hypot(*a1))
    else:
        L_px = float(1 / np.hypot(*gr.G[0]))
    L = L_px * mpp_eff * (1 - parallax_ppm * 1e-6)
    T = args.temp if args.temp is not None else s['T_ref']
    L20 = period.temperature_correct(L, T, s['scale_cte'], s['scale_T_cert'], s['grating_cte'],
                                     s['T_ref'])
    span = (tr.s[tr.used].max() - tr.s[tr.used].min()) * tr.mpp
    u_terms = []
    uL = cert.U_at(tr.X_um.min()) / 2 * np.sqrt(2 * (1 - cert.correlation))
    u_terms.append(("scale certificate over the field", uL / span * 1e6))
    n = tr.used.sum()
    u_terms.append(("tick fit", tr.rms_um * np.sqrt(12 / n) / span * 1e6))
    u_terms.append(("scale-fit model", abs(mpp_lin / mpp_eff - 1) * 1e6 / np.sqrt(3)))
    u_terms.append(("height / magnification", height_u))
    u_terms.append(("temperature", abs(s['scale_cte'] - s['grating_cte']) * s['temp_u']))
    uc = float(np.sqrt(sum(u ** 2 for _, u in u_terms)))
    print(f"KeyMaster {VERSION} QUICK (single field, no stepping)")
    print(f"  scale image {os.path.basename(scale_path)}: {n} ticks, rms {tr.rms_um:.3f} um, "
          f"{mpp_eff:.6f} um/px")
    print(f"  grating image {os.path.basename(grat_path)}: period {L_px:.4f} px")
    if same:
        print("  ! Scale and grating in one image: the gap between them changes the "
              "magnification; set [quick] parallax_ppm if it is known.")
    print(f"  Period at {s['T_ref']:.0f} C: {L20:.4f} um +/- {2 * uc * L20 * 1e-6:.4f} um "
          f"(k=2) = {2 * uc:.0f} ppm   [quick-mode estimate, not the stepped measurement]")
    for name, u in u_terms:
        print(f"    {name}: {2 * u:.0f} ppm (k=2)")
    return L20, 2 * uc


def main(argv=None):
    ap = argparse.ArgumentParser(description="KeyMaster: grating period from Keyence images "
                                             "against a certified stage micrometer.")
    ap.add_argument('--config', help="config file (default: config.ini next to this script)")
    ap.add_argument('--run', help="run table CSV (overrides [paths] run_csv)")
    ap.add_argument('--images', help="image folder (overrides [paths] images_dir)")
    ap.add_argument('--cert', help="certificate CSV (overrides [paths] cert_csv; 'none' for none)")
    ap.add_argument('--out', help="output root folder (overrides [paths] output_dir)")
    ap.add_argument('--workers', type=int, help="parallel processes for image analysis")
    ap.add_argument('--quick', nargs='+', metavar='IMAGE',
                    help="quick mode: SCALE_IMAGE [GRATING_IMAGE]")
    ap.add_argument('--keyence', nargs='+', metavar='IMAGE',
                    help="Keyence-calibration mode (no micrometer, not traceable): IMAGES or "
                         "wildcards; um/px is read from each VHX TIFF")
    ap.add_argument('--scale-mm', type=float, help="quick mode: approximate scale reading (mm)")
    ap.add_argument('--rot', type=int, default=0,
                    help="quick/Keyence mode: grating rotation (deg); 0/180 = grating x axis")
    ap.add_argument('--temp', type=float, help="quick/Keyence mode: grating temperature (C)")
    args = ap.parse_args(argv)
    if args.keyence:
        return keyence_mode(args)
    if args.quick:
        return quick(args)
    return run(args)


if __name__ == '__main__':
    main()
