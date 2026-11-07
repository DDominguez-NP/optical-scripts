# Across images: count periods, fit the period, correct it and build the uncertainty budget.
#
# Each usable stop carries the scale coordinates of one lattice point (X along the scale, Y
# across it, um), that image's lattice vectors in scale coordinates (A1, A2, um), the point's
# along-axis distance from the optical centre (d, for parallax) and the half-shift score ratios.
#
# Counting is done stop to stop within a pass, on the "effective" lattice: the true lattice, or
# a finer one when the image cannot tell a half-cell shift apart (odd-orders-only gratings).
# Lattice coordinates (n1, n2) are in true periods and may be half-integers.
#
# Fit (per grating axis, all passes pooled, one intercept per pass):
#   X - n2 * A2x = A_pass + S * n1 [+ beta * d]
# 2D lattice: S = Lambda cos(alpha)   (same lattice point tracked along a1)
# 1D grating: S = Lambda / cos(alpha) (crossings of the lines with the scale axis)

import numpy as np

HALF_BASES = {
    'none': np.eye(2),
    'half_1': np.diag([0.5, 1.0]),
    'half_2': np.diag([1.0, 0.5]),
    'centred': np.array([[0.5, 0.5], [0.5, -0.5]]),
    'half_both': np.diag([0.5, 0.5]),
}


def effective_lattice(ratios, dims, threshold):
    """Which half-cell shifts the images cannot distinguish (median score ratios)."""
    r = {k: np.nanmedian([x.get(k, np.nan) for x in ratios]) for k in ('10', '01', '11')}
    amb = {k: (v > threshold) for k, v in r.items() if np.isfinite(v)}
    if dims == 1:
        return ('half_1' if amb.get('10') else 'none'), r
    if amb.get('10') and amb.get('01'):
        return 'half_both', r
    if amb.get('11'):
        return 'centred', r
    if amb.get('10'):
        return 'half_1', r
    if amb.get('01'):
        return 'half_2', r
    return 'none', r


def count_pass(stops, dims, eff, tol=0.25):
    """Assign lattice coordinates stop to stop (stops sorted by X). Modifies the dicts."""
    stops.sort(key=lambda s: s['X'])
    M = HALF_BASES[eff]
    A = np.column_stack([np.median([s['A1'] for s in stops], axis=0),
                         np.median([s['A2'] for s in stops], axis=0)])
    if dims == 1:
        step = np.median([s['A1'][0] for s in stops]) * M[0, 0]
        m = 0
        for i, s in enumerate(stops):
            if i:
                c = (s['X'] - stops[i - 1]['X']) / step + m
                m = int(np.rint(c))
                s['count_frac'] = float(c - m)
            else:
                s['count_frac'] = 0.0
            s['n1'], s['n2'] = M[0, 0] * m, 0.0
        return
    E = A @ M
    m = np.zeros(2)
    for i, s in enumerate(stops):
        if i:
            P, Pp = np.array([s['X'], s['Y']]), np.array([stops[i - 1]['X'], stops[i - 1]['Y']])
            c = np.linalg.solve(E, P - Pp) + m
            m = np.rint(c)
            s['count_frac'] = float(np.max(np.abs(c - m)))
        else:
            s['count_frac'] = 0.0
        n = M @ m
        s['n1'], s['n2'] = float(n[0]), float(n[1])
    for s in stops:
        if s['count_frac'] > tol:
            s['flags'].append(f"period count uncertain (fraction {s['count_frac']:.2f})")


def _design(stops, passes, parallax):
    cols = [[1.0 if s['pass'] == p else 0.0 for s in stops] for p in passes]
    cols.append([s['n1'] for s in stops])
    if parallax:
        cols.append([s['d_um'] for s in stops])
    return np.array(cols).T


def fit_period(stops, parallax, outlier_sigma=3.0, x_key='X'):
    """Pooled LS fit; returns a dict (slope S, its SE, beta, residuals, used flags)."""
    use = np.array([True] * len(stops))
    y_all = np.array([s[x_key] - s['n2'] * s['A2'][0] for s in stops])
    for _ in range(len(stops)):
        st = [s for s, u in zip(stops, use) if u]
        passes = sorted({s['pass'] for s in st})
        X = _design(st, passes, parallax)
        y = y_all[use]
        if len({s['n1'] for s in st}) < 2 or len(y) <= X.shape[1]:
            return None
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        dof = len(y) - X.shape[1]
        sigma = np.sqrt(np.sum(res ** 2) / dof)
        worst = np.argmax(np.abs(res))
        if dof > 2 and abs(res[worst]) > outlier_sigma * sigma:
            idx = np.flatnonzero(use)[worst]
            use[idx] = False
            stops[idx]['flags'].append(f"rejected as outlier ({res[worst]:+.3f} um)")
            continue
        break
    cov = np.linalg.inv(X.T @ X) * sigma ** 2
    k = len(passes)
    full_res = np.full(len(stops), np.nan)
    Xall = _design(stops, passes, parallax) if all(s['pass'] in passes for s in stops) else None
    if Xall is not None:
        full_res = y_all - Xall @ coef
    else:
        full_res[use] = res
    return dict(S=float(coef[k]), se_S=float(np.sqrt(cov[k, k])),
                beta=float(coef[k + 1]) if parallax else 0.0,
                se_beta=float(np.sqrt(cov[k + 1, k + 1])) if parallax else 0.0,
                intercepts={p: float(c) for p, c in zip(passes, coef[:k])},
                sigma=float(sigma), dof=int(dof), used=use, residuals=full_res,
                parallax=parallax)


def per_pass_slopes(stops, beta, key='X'):
    out = {}
    for p in sorted({s['pass'] for s in stops}):
        st = [s for s in stops if s['pass'] == p and s.get('used')]
        n = np.array([s['n1'] for s in st])
        if len(st) < 3 or len(np.unique(n)) < 2:
            continue
        y = np.array([s[key] - s['n2'] * s['A2'][0] - beta * s['d_um'] for s in st])
        X = np.column_stack([np.ones_like(n), n])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        dof = len(y) - 2
        se = np.sqrt(np.sum(res ** 2) / dof / np.sum((n - n.mean()) ** 2)) if dof > 0 else np.nan
        out[p] = dict(S=float(coef[1]), se=float(se), n=len(st),
                      direction=st[0]['direction'], rot=st[0]['rot'])
    return out


def to_period(S, alpha_rad, dims):
    return abs(S) / np.cos(alpha_rad) if dims == 2 else abs(S) * np.cos(alpha_rad)


def temperature_correct(L, T_run, scale_cte, scale_T_cert, grating_cte, T_ref):
    L_scale = L * (1 + scale_cte * 1e-6 * (T_run - scale_T_cert))
    return L_scale / (1 + grating_cte * 1e-6 * (T_run - T_ref))


def compare_groups(slopes, key, a, b):
    ga = [v for v in slopes.values() if v[key] == a and np.isfinite(v['se'])]
    gb = [v for v in slopes.values() if v[key] == b and np.isfinite(v['se'])]
    if not ga or not gb:
        return None

    def wmean(g):
        w = np.array([1 / v['se'] ** 2 for v in g])
        x = np.array([abs(v['S']) for v in g])
        return float(np.sum(w * x) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))
    ma, ua = wmean(ga)
    mb, ub = wmean(gb)
    z = (ma - mb) / np.hypot(ua, ub)
    return dict(groups=f"{key} {a} vs {b}", mean_a=ma, u_a=ua, mean_b=mb, u_b=ub,
                difference_um=ma - mb, z=float(z), consistent=bool(abs(z) <= 2.0))


def uncertainty_budget(L20, fit, slopes, stops, cert, cfg, dims, alpha_mean, alpha_u,
                       L_alt_tick, L_alt_parallax, T_run, span_um):
    """Terms as (name, u_um at k=1, note). Lengths relative to the period L20."""
    terms = []
    rel = L20 / max(abs(fit['S']), 1e-12)
    # statistical: pooled-fit SE vs between-pass spread, whichever is larger
    se_fit = fit['se_S'] * rel
    vals = [abs(v['S']) for v in slopes.values()]
    se_pass = np.std(vals, ddof=1) / np.sqrt(len(vals)) * rel if len(vals) >= 2 else 0.0
    terms.append(("statistical (location and repeatability)", max(se_fit, se_pass),
                  f"larger of pooled-fit SE {se_fit:.2e} um and between-pass spread "
                  f"{se_pass:.2e} um"))
    # certificate: end-to-end length over the measured span
    x_lo = min(s['X'] for s in stops if s.get('used'))
    x_hi = max(s['X'] for s in stops if s.get('used'))
    u1, u2 = cert.U_at(x_lo) / 2, cert.U_at(x_hi) / 2
    rho = cert.correlation
    uL = np.sqrt(max(u1 ** 2 + u2 ** 2 - 2 * rho * u1 * u2, 0.0))
    terms.append(("scale certificate", L20 * uL / span_um,
                  f"U(k=2) {cert.U_at(x_lo):.3f}/{cert.U_at(x_hi):.3f} um at the ends, "
                  f"correlation {rho}, span {span_um / 1000:.2f} mm"
                  + ("" if cert.certified else " (NO certificate loaded: nominal positions)")))
    ui = cert.interpolation_u_um()
    terms.append(("certificate interpolation", L20 * ui * np.sqrt(2) / span_um,
                  f"leave-one-out RMS of the deviation interpolation {ui:.3f} um"))
    # temperature
    a_s, a_g = cfg['scale_cte'] * 1e-6, cfg['grating_cte'] * 1e-6
    uT = cfg['temp_u']
    u_temp = L20 * np.sqrt((abs(a_s - a_g) * uT) ** 2
                           + (cfg['scale_cte_u'] * 1e-6 * (T_run - cfg['scale_T_cert'])) ** 2
                           + (cfg['grating_cte_u'] * 1e-6 * (T_run - cfg['T_ref'])) ** 2)
    terms.append(("temperature", u_temp,
                  f"u(T) {uT} C, CTE scale {cfg['scale_cte']}+/-{cfg['scale_cte_u']}, "
                  f"grating {cfg['grating_cte']}+/-{cfg['grating_cte_u']} ppm/C, "
                  f"T_run {T_run:.2f} C"))
    terms.append(("angle", L20 * abs(np.tan(alpha_mean)) * alpha_u,
                  f"alpha {alpha_mean * 1e3:.3f} +/- {alpha_u * 1e3:.3f} mrad"))
    terms.append(("scale-fit model (linear vs set order)", abs(L_alt_tick - L20) / np.sqrt(3),
                  "half the change, rectangular"))
    # the self-calibrated gap correction is good to ~10% of itself (distortion biases the
    # in-image period); take a tenth of the uncorrected-vs-corrected change, rectangular
    terms.append(("gap parallax correction",
                  0.1 * abs(L_alt_parallax - L20) / np.sqrt(3) if L_alt_parallax is not None
                  else 0.0,
                  "10% of the correction's effect" +
                  (f"; residual regressor beta {fit['beta'] * 1e6:.0f} +/- "
                   f"{fit['se_beta'] * 1e6:.0f} ppm" if fit['parallax'] else "")))
    uc = float(np.sqrt(sum(u ** 2 for _, u, _ in terms)))
    return terms, uc
