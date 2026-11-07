# End-to-end validation of KeyMaster on synthetic runs with known periods.
#
#   python validate.py [--only NAME ...] [--workers 4] [--reuse]
#
# Each scenario renders a run with synth.py into synthetic/<name>, analyses it with
# KeyMaster.py and compares the result with the true period:
#   noise-free runs must recover the period to better than 2 ppm;
#   noisy runs must agree within their stated expanded uncertainty (k=2).

import os
import sys
import json
import argparse
import contextlib
import io

import KeyMaster
import synth

SCENARIOS = {
    'clean_edmund_odd2d': dict(profile='edmund59276_metric', grating='odd2d', noise=False,
                               passes=[('fwd', 0), ('rev', 0)], limit_ppm=2.0),
    'noisy_edmund_odd2d': dict(profile='edmund59276_metric', grating='odd2d', noise=True,
                               passes=[('fwd', 0), ('rev', 0), ('fwd', 180)]),
    'noisy_sm4_all2d_left': dict(profile='sm4', grating='all2d', noise=True, increases_to='left',
                                 passes=[('fwd', 0), ('rev', 0)],
                                 grating_centre_um=(12500.0 - 4.2, 3.3)),
    'noisy_edmund_mixed2d_xy': dict(profile='edmund59276_metric', grating='mixed2d', noise=True,
                                    passes=[('fwd', 0), ('rev', 0), ('fwd', 90), ('rev', 270)]),
    'noisy_edmund_odd1d': dict(profile='edmund59276_metric', grating='odd1d', noise=True,
                               passes=[('fwd', 0), ('rev', 0)]),
}


def run_scenario(name, spec, workers, reuse):
    out = os.path.join(KeyMaster.SCRIPT_DIR, 'synthetic', name)
    spec = dict(spec)
    limit = spec.pop('limit_ppm', None)
    cfg = synth.Synth(seed=sum(map(ord, name)), **spec)
    if not (reuse and os.path.exists(os.path.join(out, 'truth.json'))):
        print(f"[{name}] rendering {len(cfg.passes) * cfg.n_stops} images ...")
        synth.make_run(out, cfg, workers)
    truth = json.load(open(os.path.join(out, 'truth.json')))
    print(f"[{name}] analysing ...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res, out_dir = KeyMaster.run(KeyMaster.argparse.Namespace(
            config=os.path.join(out, 'config.ini'), run=None, images=None, cert=None, out=None,
            workers=workers))
    ok = True
    lines = []
    for ax in res['axes']:
        true = truth['period_x_um'] if ax['axis'] == 'x' else truth['period_y_um']
        if 'error' in ax:
            lines.append(f"  {ax['axis']}: ERROR {ax['error']}")
            ok = False
            continue
        err_ppm = (ax['period_20C_um'] / true - 1) * 1e6
        U = ax['U_k2_ppm']
        if limit is not None:
            good = abs(err_ppm) < limit
            crit = f"|error| < {limit} ppm"
        else:
            good = abs(err_ppm) <= U
            crit = f"|error| <= U = {U:.1f} ppm"
        ok &= good
        stat = [b for b in ax['budget'] if b['term'].startswith('statistical')][0]
        lines.append(f"  {ax['axis']}: measured {ax['period_20C_um']:.5f} um, true {true:.5f} um,"
                     f" error {err_ppm:+.2f} ppm ({crit}) -> {'OK' if good else 'FAIL'};"
                     f" fit sigma {ax['fit_sigma_um'] * 1e3:.1f} nm, statistical "
                     f"{stat['U_k2_ppm']:.1f} ppm (k=2), counting lattice "
                     f"{ax['effective_lattice']}, gap {ax['gap_magnification_ppm']:+.0f} ppm,"
                     f" {ax['n_used']}/{ax['n_stops']} stops")
        for c in ax['consistency']:
            lines.append(f"     {c['groups']}: z = {c['z']:+.2f}")
    if res['n_failed']:
        lines.append(f"  {res['n_failed']} images failed: " + "; ".join(res['stops_flagged'][:3]))
        ok = False
    print(f"[{name}] {'PASS' if ok else 'FAIL'}  ({res['run_time_s']:.0f} s, {out_dir})")
    print("\n".join(lines))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--reuse', action='store_true', help="reuse already rendered runs")
    a = ap.parse_args()
    names = a.only or list(SCENARIOS)
    results = {n: run_scenario(n, SCENARIOS[n], a.workers, a.reuse) for n in names}
    print()
    for n, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
