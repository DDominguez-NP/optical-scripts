# QC plots (PNG) for a KeyMaster run.

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# validated categorical order (one colour per pass, fixed order, never cycled)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
    'font.size': 10, 'legend.frameon': False,
})


def _style(p, passes):
    i = passes.index(p)
    if i >= len(SERIES):
        return dict(color=INK2, marker='.')
    return dict(color=SERIES[i], marker=MARKERS[i])


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=130)
    plt.close(fig)


def fit_and_residuals(out_dir, stops, ax_res):
    axis = ax_res['axis']
    passes = sorted({s['pass'] for s in stops})
    S = ax_res['fit']['S']
    beta = ax_res['parallax_beta_ppm'] * 1e-6
    icpt = ax_res['fit']['intercepts']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    nmin = min(s['n1'] for s in stops)
    nmax = max(s['n1'] for s in stops)
    ax.plot([nmin, nmax], [S * nmin / 1000, S * nmax / 1000], color=INK2, lw=2, zorder=1,
            label=f"fit, {abs(S):.4f} um per period along the axis")
    for p in passes:
        st = [s for s in stops if s['pass'] == p and p in icpt]
        n = np.array([s['n1'] for s in st])
        y = np.array([s['Xt'] - s['n2'] * s['A2'][0] - beta * s['d_um'] - icpt[p] for s in st])
        used = np.array([s.get('used', False) for s in st])
        sty = _style(p, passes)
        ax.plot(n[used], y[used] / 1000, ls='none', ms=8, mec=SURFACE, mew=1.5,
                label=f"pass {p}", **sty)
        if (~used).any():
            ax.plot(n[~used], y[~used] / 1000, ls='none', ms=8, mfc='none', mew=1.5,
                    color=sty['color'], marker=sty['marker'])
    ax.set_xlabel("period number n")
    ax.set_ylabel("position along the scale (mm, pass intercept removed)")
    ax.set_title(f"Grating {axis} axis: position vs period number", loc='left')
    ax.legend()
    _save(fig, out_dir, f"fit_{axis}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color=INK2, lw=1)
    for p in passes:
        st = [s for s in stops if s['pass'] == p]
        X = np.array([s['X'] for s in st]) / 1000
        r = np.array([s.get('residual_um', np.nan) for s in st]) * 1000
        used = np.array([s.get('used', False) for s in st])
        sty = _style(p, passes)
        ax.plot(X[used], r[used], ls='-', lw=1, ms=8, mec=SURFACE, mew=1.5,
                label=f"pass {p}", **sty)
        if (~used).any():
            ax.plot(X[~used], r[~used], ls='none', ms=8, mfc='none', mew=1.5,
                    color=sty['color'], marker=sty['marker'])
    ax.set_xlabel("position along the scale (mm)")
    ax.set_ylabel("fit residual (nm)")
    ax.set_title(f"Grating {axis} axis: residuals vs position (open = rejected)", loc='left')
    ax.legend()
    _save(fig, out_dir, f"residuals_{axis}.png")


def pass_slopes(out_dir, ax_res):
    pp = ax_res['per_pass']
    if not pp:
        return
    axis = ax_res['axis']
    keys = list(pp)
    L = np.array([pp[k]['period_um'] for k in keys])
    u = np.array([pp[k]['se'] * pp[k]['period_um'] / abs(pp[k]['S']) for k in keys])
    L20 = ax_res['period_20C_um']
    LT = ax_res['period_at_T_um']
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    ppm = (L / LT - 1) * 1e6
    ax.axhspan(-ax_res['U_k2_ppm'], ax_res['U_k2_ppm'], color=GRID, alpha=0.6, lw=0,
               label=f"pooled result +/- U (k=2), {ax_res['U_k2_ppm']:.1f} ppm")
    ax.axhline(0, color=INK2, lw=1)
    for i, k in enumerate(keys):
        sty = _style(k, keys)
        ax.errorbar(x[i], ppm[i], yerr=2 * u[i] / LT * 1e6, ls='none', ms=8, capsize=4,
                    color=sty['color'], marker=sty['marker'], mec=SURFACE)
    ax.set_xticks(x, [f"pass {k}\n{pp[k]['direction']} {pp[k]['rot']} deg" for k in keys])
    ax.set_ylabel("per-pass period vs pooled (ppm), bars k=2")
    ax.set_title(f"Grating {axis} axis: per-pass periods (pooled {L20:.5f} um at 20 C)",
                 loc='left')
    ax.legend(loc='upper right')
    _save(fig, out_dir, f"pass_periods_{axis}.png")


def mpp_stability(out_dir, records):
    ok = [r for r in records if r['ok']]
    if not ok:
        return
    passes = sorted({r['pass'] for r in ok})
    fig, ax = plt.subplots(figsize=(7, 4))
    i = 0
    for p in passes:
        st = [r for r in ok if r['pass'] == p]
        idx = np.arange(i, i + len(st))
        i += len(st)
        sty = _style(p, passes)
        ax.plot(idx, [r['mpp_um'] for r in st], ls='-', lw=1, ms=8, mec=SURFACE, mew=1.5,
                label=f"pass {p}", **sty)
    ax.set_xlabel("image (run order)")
    ax.set_ylabel("um per pixel (from ticks)")
    ax.set_title("Pixel scale stability", loc='left')
    ax.ticklabel_format(axis='y', useOffset=False)
    if len(passes) > 1:
        ax.legend()
    _save(fig, out_dir, "um_per_px.png")


def tick_residuals(out_dir, records):
    ok = [r for r in records if r['ok']]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    allr = []
    for r in ok:
        t = r['_ticks']
        u = t['used']
        ax.plot(t['s'][u] * r['mpp_um'] / 1000, t['resid'][u] * 1000, ls='none', marker='.',
                ms=3, color=SERIES[0], alpha=0.25)
        allr.append(t['resid'][u])
    allr = np.concatenate(allr) * 1000
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("position in the image along the scale axis, from the anchor (mm)")
    ax.set_ylabel("tick-fit residual (nm)")
    ax.set_title(f"Tick-fit residuals, all images (RMS {np.sqrt(np.mean(allr ** 2)):.0f} nm)",
                 loc='left')
    _save(fig, out_dir, "tick_residuals.png")


def annotated_image(out_dir, rec, images_dir):
    path = os.path.join(images_dir, rec['file'])
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        import tifffile
        img = tifffile.imread(path).astype(float)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path)).astype(float)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    t = rec['_ticks']
    H, W = img.shape
    lo, hi = np.percentile(img, [0.5, 99.8])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap='gray', interpolation='lanczos')
    ax.grid(False)
    c, s_ = np.cos(t['theta']), np.sin(t['theta'])
    xs = np.array([0, W - 1])
    ax.plot(xs, t['y0'] + np.tan(t['theta']) * (xs - t['xc']), color=SERIES[0], lw=1,
            label="baseline (scale axis)")
    sx = t['xc'] + t['s'] * c
    sy = t['y0'] + t['s'] * s_
    u = t['used']
    ax.plot(sx[u], sy[u], ls='none', marker='|', ms=12, mew=1.5, color=SERIES[2],
            label=f"ticks used ({u.sum()})")
    if (~u).any():
        ax.plot(sx[~u], sy[~u], ls='none', marker='|', ms=12, mew=1.5, color=SERIES[7],
                label="ticks rejected / excluded")
    q = t['q']
    ax.plot(q[0], q[1], marker='+', ms=18, mew=2.5, color=SERIES[1], ls='none',
            label="tracked lattice point")
    for a, col in zip(t['A'], (SERIES[1], SERIES[4])):
        ax.annotate('', xy=(q[0] + a[0], q[1] + a[1]), xytext=(q[0], q[1]),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.plot(t['xc'], t['y0'], marker='o', ms=8, mfc='none', mew=2, color=SERIES[3], ls='none',
            label="anchor (baseline at image centre column)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f"{rec['file']}: X = {rec['X'] / 1000:.4f} mm, {rec['mpp_um']:.5f} um/px",
                 loc='left')
    ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor=SURFACE)
    _save(fig, out_dir, "example_annotated.png")


def make_all(out_dir, records, axes, images_dir, s):
    ok = [r for r in records if r['ok']]
    for a in axes:
        if 'error' in a:
            continue
        st = [r for r in ok if r['axis'] == a['axis'] and 'n1' in r]
        fit_and_residuals(out_dir, st, a)
        pass_slopes(out_dir, a)
    mpp_stability(out_dir, records)
    tick_residuals(out_dir, records)
    if ok:
        mid = ok[len(ok) // 2]
        annotated_image(out_dir, mid, images_dir)
