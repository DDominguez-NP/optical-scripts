# ArgusEye: diffraction-angle measurement of a DOE by substitution against a reference grating.
#
# Given a camera image of the reference grating and one of the sample, both taken through the
# same Fourier lens and camera without changes in between, this script:
#   1. Finds, indexes and centroids the spot grids in both images (spots.py).
#   2. Calibrates the camera in angle from the reference, whose period is known (lattice.py):
#      focal length in pixels, rotation and Brown-Conrady lens distortion (k1 by default).
#   3. Fits the sample's spot lattice in direction cosines with that calibration, giving the
#      sample period along both grating axes.
#   4. Reports, per axis, the angle between the -1 and +1 orders, the angle between neighbouring
#      spots, the full angle between the outermost spots, and their uncertainties.
#   5. Writes results.json, per-spot CSVs and plots to a time-stamped output folder.
#
# Spots cut by the sensor edge or saturated are kept out of the fits. Angles are reported at the
# wavelength in config.ini; the camera scale cancels, so the laser's actual wavelength does not
# enter as long as both images were taken with the same laser.
#
# File paths and options are read from config.ini, located next to this script, and can be
# overridden on the command line (python ArgusEye.py --help).

import os
import sys
import csv
import glob
import json
import argparse
import configparser
from dataclasses import asdict
from datetime import datetime

import numpy as np

import sources
import spots
import lattice

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_configuration(config_file=os.path.join(SCRIPT_DIR, "config.ini")):
    if not os.path.exists(config_file):
        print(f"[!] Configuration file not found: {config_file}")
        sys.exit(1)
    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    config.read(config_file)
    return config


def _get_float(section, key):
    """A float from config, or None when the key is missing or blank."""
    value = section.get(key, fallback='').strip()
    return float(value) if value else None


def _resolve(path):
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path))


def _split_paths(text):
    """Comma-separated paths; entries with * or ? are wildcard patterns, expanded in sorted order."""
    paths = []
    for p in (p.strip() for p in text.split(',')):
        if not p:
            continue
        path = _resolve(p)
        if glob.has_magic(path):
            matches = sorted(glob.glob(path))
            if not matches:
                raise FileNotFoundError(f"No files match {path}")
            paths.extend(matches)
        elif not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        else:
            paths.append(path)
    return paths


def read_period_file(path):
    """
    Reference period from a KeyMaster grating_period.json. KeyMaster's x is ArgusEye's x (the
    reference's nx direction); while y is not measured, x is used for y with the same uncertainty
    (fully correlated). Returns period_x, period_y (um), u_x, u_y (relative) and the file contents.
    """
    with open(path) as f:
        data = json.load(f)
    period_x, u_x = data['period_x_um'], data['u_period_x_ppm'] * 1e-6
    if data.get('y_measured') and data.get('period_y_um') is not None:
        period_y, u_y = data['period_y_um'], data['u_period_y_ppm'] * 1e-6
    else:
        period_y, u_y = period_x, u_x
    return period_x, period_y, u_x, u_y, data


def reference_period(ref_cfg, nx, ny, wavelength_nm):
    """
    lambda / period of the reference along x and y, their relative uncertainties, where they
    came from, and whether the result is interim (reference not traceably measured).
    Priority: KeyMaster period_file, then period_um from config, then the nominal separation.
    """
    lam_um = wavelength_nm * 1e-3
    path = ref_cfg.get('period_file', fallback='').strip()
    if path:
        path = _resolve(path)
        period_x, period_y, u_x, u_y, data = read_period_file(path)
        interim = not data.get('traceable', False)
        source = (f"KeyMaster period {period_x:.3f} x {period_y:.3f} um"
                  + ("" if data.get('y_measured') else " (y = x, y not measured)")
                  + f", {data.get('length_reference', 'length reference not given')}")
        return dict(sin_x=lam_um / period_x, sin_y=lam_um / period_y, u_rel=np.array([u_x, u_y]),
                    source=source, interim=interim, period_file=path,
                    interim_reason="reference period is not traceable (KeyMaster traceable = false)"
                    if interim else None)

    u_ppm = _get_float(ref_cfg, 'period_uncertainty_ppm') or 0.0
    period_x = _get_float(ref_cfg, 'period_um')
    if period_x:
        period_y = _get_float(ref_cfg, 'period_y_um') or period_x
        return dict(sin_x=lam_um / period_x, sin_y=lam_um / period_y,
                    u_rel=np.array([u_ppm, u_ppm]) * 1e-6,
                    source=f"period {period_x} x {period_y} um from config.ini",
                    interim=False, period_file=None, interim_reason=None)

    sep_x = _get_float(ref_cfg, 'separation_deg')
    if sep_x is None:
        raise ValueError("[reference] needs period_file, period_um or separation_deg")
    sep_y = _get_float(ref_cfg, 'separation_y_deg') or sep_x

    def sin_step(sep, count):
        # Neighbouring spots are orders -1/+1 for an even count, 0/+1 for an odd count
        return np.sin(np.radians(sep / 2 if count % 2 == 0 else sep))
    return dict(sin_x=sin_step(sep_x, nx), sin_y=sin_step(sep_y, ny),
                u_rel=np.array([u_ppm, u_ppm]) * 1e-6,
                source=f"nominal separation {sep_x} x {sep_y} deg", interim=True, period_file=None,
                interim_reason="reference period is the datasheet nominal, not a measurement")


def analyse(ref_frame, sample_frame, config):
    cam, det = config['camera'], config['detection']
    ref_cfg, smp_cfg = config['reference'], config['sample']
    wavelength_nm = config['wavelength'].getfloat('wavelength_nm')
    det_args = dict(threshold_sigma=det.getfloat('threshold_sigma', 8.0),
                    min_area=det.getint('min_area_px', 20),
                    saturation_fraction=det.getfloat('saturation_fraction', 0.98))

    rnx, rny = ref_cfg.getint('nx'), ref_cfg.getint('ny')
    snx, sny = smp_cfg.getint('nx'), smp_cfg.getint('ny')
    ref_pat = spots.measure_pattern(ref_frame, rnx, rny, **det_args)
    smp_pat = spots.measure_pattern(sample_frame, snx, sny, **det_args)

    ref_info = reference_period(ref_cfg, rnx, rny, wavelength_nm)
    sin_x, sin_y = ref_info['sin_x'], ref_info['sin_y']
    F0 = cam.getfloat('focal_length_mm') * 1e3 / cam.getfloat('pixel_pitch_um')
    model = lattice.DistortionModel.parse(det.get('distortion', 'k1'))
    ref_fit = lattice.fit_reference(ref_pat, sin_x, sin_y, F0, model)
    smp_fit = lattice.fit_sample(smp_pat, ref_fit)
    # Self-check: measure the reference as if it were a sample; it should return its own period
    ref_check = lattice.fit_sample(ref_pat, ref_fit)

    # Distortion-model sensitivity: redo the measurement with the tangential terms toggled. The
    # shift counts only if that model also fits the reference (an inadequate model is not a
    # plausible alternative); it is taken as the half-width of a rectangular distribution.
    alt_model = model.alternative()
    alternative = dict(model=alt_model.name, counted=False, shift_ppm=None, rms_px=None)
    u_model = np.zeros(2)
    try:
        alt_ref = lattice.fit_reference(ref_pat, sin_x, sin_y, F0, alt_model, leave_one_out=False)
        shift = lattice.fit_sample(smp_pat, alt_ref).mags / smp_fit.mags - 1
        alternative.update(shift_ppm=shift * 1e6, rms_px=_rms(alt_ref.residuals),
                           counted=_rms(alt_ref.residuals) <= lattice.ADEQUATE_RMS_RATIO * _rms(ref_fit.residuals))
        if alternative['counted']:
            u_model = np.abs(shift) / np.sqrt(3)
    except RuntimeError:
        pass

    axes = {}
    for a, (name, g, idx, count) in enumerate((('X', smp_fit.g1, slice(2, 4), snx),
                                               ('Y', smp_fit.g2, slice(4, 6), sny))):
        mag = np.hypot(*g)
        u_fit = lattice.magnitude_uncertainty(g, smp_fit.cov[idx, idx]) / mag
        u_cal = smp_fit.u_g_calib[a] / mag
        u_ref = float(ref_info['u_rel'][a])
        u_rel = np.sqrt(u_fit ** 2 + u_cal ** 2 + u_model[a] ** 2 + u_ref ** 2)
        res = lattice.axis_result(name, count, mag, u_rel, wavelength_nm)
        axes[name] = dict(asdict(res), u_rel_fit=u_fit, u_rel_calibration=u_cal,
                          u_rel_model=float(u_model[a]), u_rel_reference=u_ref)

    spec_sep = _get_float(smp_cfg, 'spec_separation_deg')
    spec_tol = _get_float(smp_cfg, 'spec_tolerance_deg')
    if spec_sep is not None and spec_tol is not None:
        for a in axes.values():
            a['spec_deviation_deg'] = a['adjacent_deg'] - spec_sep
            a['pass'] = bool(abs(a['spec_deviation_deg']) <= spec_tol)

    # Zero-order leak: should sit on the fitted pattern centre (direction 0, 0)
    zero = None
    if not (snx % 2 and sny % 2):
        pitch = min(np.hypot(*smp_pat.a1), np.hypot(*smp_pat.a2))
        zero = spots.measure_point(smp_pat.image, *smp_fit.c, smp_pat.spot_sigma, smp_pat.noise, pitch)
        zero['offset_px'] = [zero['x'] - smp_fit.c[0], zero['y'] - smp_fit.c[1]]

    def recovered(g, nominal):
        return dict(sin_step=float(np.hypot(*g)), ppm=float((np.hypot(*g) / nominal - 1) * 1e6))

    g1, g2 = smp_fit.g1, smp_fit.g2
    diagnostics = dict(
        reference_self_check_x=recovered(ref_check.g1, sin_x),
        reference_self_check_y=recovered(ref_check.g2, sin_y),
        sample_rotation_deg=float(np.degrees(np.arctan2(g1[1], g1[0]))),
        reference_rotation_deg=float(np.degrees(ref_fit.phi)),
        sample_axes_angle_deg=float(np.degrees(np.arccos(g1 @ g2 / np.hypot(*g1) / np.hypot(*g2)))),
        sample_zero_order_direction_mrad=(smp_fit.zero_dir * 1e3).tolist(),
        sample_spots_outside_calibrated_field=smp_fit.n_extrapolated,
        reference_field_deg=np.degrees(np.arctan(ref_fit.t_extent)),
        sample_field_deg=np.degrees(np.arctan(smp_fit.t_extent)),
        reference_extrapolation_rms_px=ref_fit.extrapolation_rms,
        distortion_alternative=alternative,
        reference_leave_one_out_rms_px=ref_fit.loo_rms,
        centroid_method_rms_diff_px=dict(reference=_method_diff(ref_pat), sample=_method_diff(smp_pat)),
        zero_order=zero,
    )

    return dict(wavelength_nm=wavelength_nm, reference_source=ref_info['source'],
                reference_info=ref_info,
                reference_sin_steps=[sin_x, sin_y], ref_pattern=ref_pat, smp_pattern=smp_pat,
                ref_fit=ref_fit, smp_fit=smp_fit, axes=axes, diagnostics=diagnostics)


def _method_diff(pattern):
    d = [(s.x - s.x_gauss) ** 2 + (s.y - s.y_gauss) ** 2 for s in pattern.used
         if np.isfinite(s.x_gauss)]
    return float(np.sqrt(np.mean(d))) if d else None


def _rms(res):
    return float(np.sqrt((res ** 2).sum(axis=1).mean()))


# --- Output ---------------------------------------------------------------------------------

def print_summary(r, ref_name, smp_name, ref_frame, smp_frame):
    rf, sf, rp, sp = r['ref_fit'], r['smp_fit'], r['ref_pattern'], r['smp_pattern']
    pitch_um = r['pixel_pitch_um']
    print(f"\nArgus  {datetime.now():%Y-%m-%d %H:%M}   angles at {r['wavelength_nm']} nm (air)")
    if r['reference_info']['interim']:
        print(f"*** INTERIM RESULT: {r['reference_info']['interim_reason']} ***")
    print(f"\nReference  {ref_name}  {rp.nx} x {rp.ny}   ({r['reference_source']})")
    print(f"  spots used {len(rp.used)}/{rp.nx * rp.ny}   fit rms {_rms(rf.residuals):.3f} px   "
          f"chi2_red {rf.chi2_red:.1f}")
    cam = rf.camera
    print(f"  focal length {rf.F:.1f} px = {rf.F * pitch_um * 1e-3:.3f} mm   "
          f"rotation {np.degrees(rf.phi):+.3f} deg   leave-one-out rms {rf.loo_rms:.3f} px")
    coefs = [f"k{i + 1} {v:+.3f}" for i, v in enumerate(cam.k_opencv)]
    if cam.model.tangential:
        coefs += [f"p1 {cam.p_opencv[0]:+.4f}", f"p2 {cam.p_opencv[1]:+.4f}"]
    print(f"  distortion (Brown-Conrady, OpenCV convention): {', '.join(coefs)}   "
          f"principal point ({cam.pp[0]:.0f}, {cam.pp[1]:.0f}) px")
    print(f"  reference zero order is ({cam.axis[0] * 1e3:+.2f}, {cam.axis[1] * 1e3:+.2f}) mrad "
          "from the lens axis")
    print(f"\nSample  {smp_name}  {sp.nx} x {sp.ny}")
    excluded = [s for s in sp.spots if not s.used]
    print(f"  spots used {len(sp.used)}/{sp.nx * sp.ny}"
          + (f"   excluded {len(excluded)} ({', '.join(sorted({s.note for s in excluded}))})" if excluded else "")
          + (f"   missing {len(sp.missing)}" if sp.missing else "")
          + f"   fit rms {_rms(sf.residuals):.3f} px   chi2_red {sf.chi2_red:.1f}")

    ax = r['axes']
    print(f"\n  {'':34s}{'X':>22s}{'Y':>22s}")

    def row(label, key, ukey, fmt='{:.5f}'):
        cells = [f"{fmt.format(ax[a][key])} +/- {fmt.format(ax[a][ukey])}" for a in ('X', 'Y')]
        print(f"  {label:34s}{cells[0]:>22s}{cells[1]:>22s}")

    row('separation -1 to +1 (deg)', 'sep_pm1_deg', 'u_sep_pm1_deg')
    if any(ax[a]['count'] % 2 for a in ax):
        row('neighbouring-spot separation (deg)', 'adjacent_deg', 'u_adjacent_deg')
    print(f"  {'full angle, outermost spots (deg)':34s}"
          + ''.join(f"{'{:.4f} +/- {:.4f}'.format(ax[a]['full_deg'], ax[a]['u_full_deg']):>22s}" for a in ('X', 'Y'))
          + f"   (orders +/-{ax['X']['outer_order']} x +/-{ax['Y']['outer_order']})")
    print(f"  {'period (um)':34s}" + ''.join(f"{ax[a]['period_um']:>22.4f}" for a in ('X', 'Y')))
    print(f"  {'uncertainty (ppm, k=1)':34s}" + ''.join(f"{ax[a]['u_rel'] * 1e6:>22.0f}" for a in ('X', 'Y')))
    print(f"  {'  of which: sample fit':34s}" + ''.join(f"{ax[a]['u_rel_fit'] * 1e6:>22.0f}" for a in ('X', 'Y')))
    print(f"  {'            camera calibration':34s}" + ''.join(f"{ax[a]['u_rel_calibration'] * 1e6:>22.0f}" for a in ('X', 'Y')))
    print(f"  {'            distortion model':34s}" + ''.join(f"{ax[a]['u_rel_model'] * 1e6:>22.0f}" for a in ('X', 'Y')))
    print(f"  {'            reference period':34s}" + ''.join(
        f"{(format(ax[a]['u_rel_reference'] * 1e6, '.0f') if ax[a]['u_rel_reference'] else 'not set'):>22}"
        for a in ('X', 'Y')))
    if 'pass' in ax['X']:
        print(f"  {'spec (neighbouring spots)':34s}" + ''.join(
            f"{('PASS' if ax[a]['pass'] else 'FAIL') + ' ({:+.5f})'.format(ax[a]['spec_deviation_deg']):>22s}"
            for a in ('X', 'Y')))
        if r['reference_info']['interim']:
            print("  (interim: pass/fail is against the nominal; the reference uncertainty is not yet "
                  "small enough to decide it)")
    else:
        print("  no spec set for this sample: pass/fail not evaluated")

    clipped_outer = any(not s.used and abs(s.m) == ax['X']['outer_order'] for s in sp.spots) or \
        any(abs(m) == ax['X']['outer_order'] for m, _ in sp.missing)
    clipped_outer_y = any(not s.used and abs(s.n) == ax['Y']['outer_order'] for s in sp.spots) or \
        any(abs(n) == ax['Y']['outer_order'] for _, n in sp.missing)
    if clipped_outer or clipped_outer_y:
        which = ' and '.join(a for a, f in (('X', clipped_outer), ('Y', clipped_outer_y)) if f)
        print(f"  note: outermost spots in {which} were not measured; the full angle there is "
              "computed from the fitted period")

    d = r['diagnostics']
    print("\nDiagnostics")
    print(f"  reference self-check: X {d['reference_self_check_x']['ppm']:+.0f} ppm, "
          f"Y {d['reference_self_check_y']['ppm']:+.0f} ppm (recovered vs input period)")
    alt = d['distortion_alternative']
    if alt['shift_ppm'] is not None:
        print(f"  distortion model {alt['model']} instead moves the result by X {alt['shift_ppm'][0]:+.0f} ppm, "
              f"Y {alt['shift_ppm'][1]:+.0f} ppm" + ("" if alt['counted'] else
              f" (not counted: its fit rms {alt['rms_px']:.3f} px is too poor)"))
    rfd, sfd = d['reference_field_deg'], d['sample_field_deg']
    print(f"  field reached: sample X +/-{sfd[0]:.2f}, Y +/-{sfd[1]:.2f} deg; reference calibrated "
          f"X +/-{rfd[0]:.2f}, Y +/-{rfd[1]:.2f} deg")
    if np.isfinite(d['reference_extrapolation_rms_px']):
        print(f"  extrapolation check: calibrated without the reference's outer ring, the model "
              f"predicts it to {d['reference_extrapolation_rms_px']:.3f} px rms")
    print(f"  sample rotation {d['sample_rotation_deg']:+.3f} deg (reference {d['reference_rotation_deg']:+.3f} deg), "
          f"sample axes at {d['sample_axes_angle_deg']:.4f} deg")
    zd = d['sample_zero_order_direction_mrad']
    print(f"  sample zero order is ({zd[0]:+.3f}, {zd[1]:+.3f}) mrad from the reference's")
    if d['sample_spots_outside_calibrated_field']:
        print(f"  {d['sample_spots_outside_calibrated_field']} sample spots lie outside the field the "
              "reference calibrated (distortion extrapolated there)")
    cm = d['centroid_method_rms_diff_px']
    print(f"  windowed vs Gaussian centroid rms difference: reference {cm['reference']:.3f} px, "
          f"sample {cm['sample']:.3f} px")
    if d['zero_order']:
        z = d['zero_order']
        print(f"  zero-order leak: SNR {z['snr']:.0f}, offset from fitted centre "
              f"({z['offset_px'][0]:+.2f}, {z['offset_px'][1]:+.2f}) px")
    for label, frame, pat in (('reference', ref_frame, rp), ('sample', smp_frame, sp)):
        notes = frame.notes + pat.notes + ([f"{pat.hot_pixels} hot pixels replaced"] if pat.hot_pixels else [])
        for n in notes:
            print(f"  {label}: {n}")
    for label, frame in (('reference', ref_frame), ('sample', smp_frame)):
        if not frame.dark_subtracted:
            print(f"  {label}: no dark frame; background estimated from the image")


def write_spot_csv(path, pattern, residuals):
    res = iter(residuals)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['m', 'n', 'x_px', 'y_px', 'x_err_px', 'y_err_px', 'x_gauss_px', 'y_gauss_px',
                    'sigma_px', 'peak', 'flux', 'snr', 'saturated', 'clipped', 'used',
                    'resid_x_px', 'resid_y_px', 'note'])
        for s in pattern.spots:
            dx, dy = next(res) if s.used else (np.nan, np.nan)
            w.writerow([s.m, s.n] + [f"{v:.4f}" for v in (s.x, s.y, s.x_err, s.y_err, s.x_gauss,
                                                             s.y_gauss, s.sigma_px)]
                       + [f"{s.peak:.1f}", f"{s.flux:.0f}", f"{s.snr:.1f}", s.saturated, s.clipped,
                          s.used, f"{dx:.4f}", f"{dy:.4f}", s.note])


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def write_json(path, r, ref_frame, smp_frame, names):
    rf, sf = r['ref_fit'], r['smp_fit']

    def frame_info(frame):
        return dict(paths=frame.paths, n_frames=frame.n_frames, bit_depth=frame.bit_depth,
                    timestamp=frame.timestamp, wavelength_nm=frame.wavelength_nm,
                    environment=asdict(frame.env) if frame.env else None,
                    dark_subtracted=frame.dark_subtracted, notes=frame.notes)

    out = dict(
        created=datetime.now(), wavelength_nm=r['wavelength_nm'], wavelength_medium='air',
        interim=r['reference_info']['interim'], interim_reason=r['reference_info']['interim_reason'],
        reference=dict(name=names[0], source=r['reference_source'],
                       period_file=r['reference_info']['period_file'],
                       u_rel_period=r['reference_info']['u_rel'],
                       sin_steps=r['reference_sin_steps'], frame=frame_info(ref_frame),
                       fit=dict(F_px=rf.F, u_F_px=rf.u_F, focal_length_mm=rf.F * r['pixel_pitch_um'] * 1e-3,
                                center_px=rf.c, rotation_deg=np.degrees(rf.phi),
                                distortion=dict(model='Brown-Conrady (OpenCV convention)',
                                                terms=rf.model.name, k=rf.camera.k_opencv,
                                                p=rf.camera.p_opencv if rf.model.tangential else None,
                                                principal_point_px=rf.camera.pp,
                                                zero_order_from_lens_axis_mrad=rf.camera.axis * 1e3),
                                chi2_red=rf.chi2_red, rms_px=_rms(rf.residuals),
                                leave_one_out_rms_px=rf.loo_rms,
                                extrapolation_rms_px=rf.extrapolation_rms,
                                spots_used=len(r['ref_pattern'].used))),
        sample=dict(name=names[1], frame=frame_info(smp_frame),
                    fit=dict(center_px=sf.c, zero_order_direction=sf.zero_dir, g1=sf.g1, g2=sf.g2,
                             chi2_red=sf.chi2_red,
                             rms_px=_rms(sf.residuals), spots_used=len(r['smp_pattern'].used),
                             missing_orders=r['smp_pattern'].missing),
                    axes=r['axes']),
        diagnostics=r['diagnostics'],
    )
    with open(path, 'w') as f:
        json.dump(_jsonable(out), f, indent=2)


def make_plots(out_dir, r, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for tag, pat, fit, name in (('reference', r['ref_pattern'], r['ref_fit'], names[0]),
                                ('sample', r['smp_pattern'], r['smp_fit'], names[1])):
        fig, ax = plt.subplots(figsize=(10, 8))
        img = np.clip(pat.resid, 0, None)
        ax.imshow(np.sqrt(img), cmap='gray', origin='upper')
        for s in pat.spots:
            color = 'lime' if s.used else 'red'
            ax.plot(s.x, s.y, '+' if s.used else 'x', color=color, ms=10)
            ax.annotate(f"{s.m},{s.n}", (s.x, s.y), xytext=(8, -8), textcoords='offset points',
                        color=color, fontsize=7)
        ax.plot(*fit.c, 'c+', ms=16)
        z = r['diagnostics']['zero_order']
        if tag == 'sample' and z:
            ax.plot(z['x'], z['y'], 'o', mfc='none', mec='orange', ms=12)
        ax.set_title(f"{name}: green = used, red = excluded, cyan = fitted centre")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overlay_{tag}.png"), dpi=120)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pat, fit, name in ((axs[0], r['ref_pattern'], r['ref_fit'], names[0]),
                               (axs[1], r['smp_pattern'], r['smp_fit'], names[1])):
        pos = np.array([[s.x, s.y] for s in pat.used])
        res = fit.residuals
        scale = 0.25 * np.ptp(pos[:, 0]) / max(np.abs(res).max(), 1e-9)
        ax.quiver(pos[:, 0], pos[:, 1], res[:, 0], res[:, 1], angles='xy', scale_units='xy',
                  scale=1 / scale, color='C3')
        for (x, y), (dx, dy) in zip(pos, res):
            ax.annotate(f"{np.hypot(dx, dy):.3f}", (x, y), xytext=(4, 4), textcoords='offset points',
                        fontsize=6)
        ax.set_xlim(0, pat.image.shape[1])
        ax.set_ylim(pat.image.shape[0], 0)
        ax.set_aspect('equal')
        ax.set_title(f"{name} residuals, model - measured (px), rms {_rms(res):.3f} px; arrows x{scale:.0f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residuals.png"), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Argus DOE diffraction-angle measurement")
    parser.add_argument('--config', default=os.path.join(SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--ref', nargs='+', help="reference image file(s); averaged")
    parser.add_argument('--sample', nargs='+', help="sample image file(s); averaged")
    parser.add_argument('--ref-dark', nargs='+', help="dark frame(s) for the reference")
    parser.add_argument('--sample-dark', nargs='+', help="dark frame(s) for the sample")
    parser.add_argument('--out', help="output folder (default: from config)")
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    config = load_configuration(args.config)
    paths = config['paths']
    def pick(cli, key):
        return _split_paths(','.join(cli) if cli else paths.get(key, ''))
    ref_files = pick(args.ref, 'reference_images')
    smp_files = pick(args.sample, 'sample_images')
    ref_dark = pick(args.ref_dark, 'reference_darks')
    smp_dark = pick(args.sample_dark, 'sample_darks')
    if not ref_files or not smp_files:
        parser.error("reference and sample images are required (command line or [paths] in config.ini)")

    cam = config['camera']
    raw_format = sources.RawFormat(width=cam.getint('raw_width', 1280),
                                   height=cam.getint('raw_height', 1024),
                                   pixel_format=cam.get('raw_pixel_format', 'auto'))
    ref_frame = sources.load_frame(ref_files, ref_dark or None, raw_format)
    smp_frame = sources.load_frame(smp_files, smp_dark or None, raw_format)
    names = (config['reference'].get('name', 'reference'), config['sample'].get('name', 'sample'))

    r = analyse(ref_frame, smp_frame, config)
    r['pixel_pitch_um'] = config['camera'].getfloat('pixel_pitch_um')
    print_summary(r, *names, ref_frame, smp_frame)

    out_root = args.out or _resolve(paths.get('output_dir', 'output'))
    out_dir = os.path.join(out_root, f"{datetime.now():%Y%m%d_%H%M%S}_{names[1]}")
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, 'results.json'), r, ref_frame, smp_frame, names)
    write_spot_csv(os.path.join(out_dir, 'spots_reference.csv'), r['ref_pattern'], r['ref_fit'].residuals)
    write_spot_csv(os.path.join(out_dir, 'spots_sample.csv'), r['smp_pattern'], r['smp_fit'].residuals)
    if not args.no_plots:
        make_plots(out_dir, r, names)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
