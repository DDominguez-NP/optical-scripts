# Argus – DOE diffraction-angle metrology station (Neurophos)

Context handoff from the Claude (Cowork) design conversation, 2026-11-03. Owner: Daniel Dominguez.

## Goal
Measure the diffraction angle of 1310 nm 9 × 5-spot DOEs.
- Nominal spot spacing 0.66° (both axes), tolerance ±0.001° (≈ ±1500 ppm).
- Measurement target: ≤ 150 ppm (k = 2) on the angle; reference-grating calibration target ≤ 50 ppm (k = 2).
- DOE period ≈ 113.7 µm. DOE orders: m_x = −4..+4 (9 spots), m_y = −2..+2 (5 spots), zero order present.

## Method (substitution with a Fourier lens)
- Collimated 1310 nm beam → grating at the front focal plane of the lens → camera at the back focal plane.
- Spot position x = f · tan θ. Orders are separated by sin θ_m = mλ/Λ.
- A reference grating (Holo/Or, Λ ≈ 242.12 µm, 1D, 8 spots at odd orders ±1, ±3, ±5, ±7, first order ≈ 0.31°) is swapped in A–B–A with the DOE on a kinematic base. The ratio of spot spacings gives Λ_DOE / Λ_ref, so the camera scale (f / pixel) cancels.
- Result: θ_DOE from Λ_DOE = Λ_ref × (ratio), sin θ = λ/Λ, with λ from the wavemeter.
- 2D affine fit of the full spot grid (pitch, rotation, centre); camera roll ±3° is fine.
- A–B–A swaps, ~5 cycles; Allan deviation picks frames per reading.
- Air-index (Ciddor) correction with T, P, RH; λ is reported in vacuum by the meter.

## Hardware
- Source: Thorlabs S3FC1310 (DFB, 1.5 mW, FC/PC), 99:1 tap TN1310R1F1, VOA50-FC.
- Wavelength: Bristol 771A-NIR laser spectrum analyzer (1% tap port).
- Collimator: Thorlabs F220FC-C (f = 11.23 mm) in LMR1.
- Fourier lens: Thorlabs AC254-060-C, f = 60 mm; tube SM1V05 + SM1L05/L10 + SM1A9 (C-mount).
- Camera: LUCID Triton TRI013S-WC (Sony IMX990 InGaAs, 1280 × 1024, 5 µm pixels, GigE, Arena SDK). Use 12/16-bit mono, fixed gain/exposure, no auto features, jumbo frames.
- Grating holders: post + gimbal on KB kinematic base; Newport CONEX-LDS autocollimator via pellicle for tilt checks (not yet received).

## Expected image content
- DOE: 45 spots on a ~5.5 × 2.8 mm grid; spacing ≈ f·tan(0.66°) ≈ 0.691 mm ≈ 138 px.
- Reference: 8 spots in a row, spacing between adjacent lit orders ≈ 2 × 0.325 mm ≈ 130 px (orders ±1, ±3, ±5, ±7 at 0.325 mm × m).
- Centroid target ≈ 0.03–0.04 px per spot.

## Analysis script requirements (first task)
1. Load camera frames (Arena-saved TIFF/PNG/raw, 16-bit). Stack and average N frames, with dark/background subtraction.
2. Detect spots (threshold + connected components), then refine centroids sub-pixel:
   intensity-weighted centroid in an ROI with background removal, or a 2D Gaussian fit. Report per-spot SNR and saturation flags.
3. Index spots to orders (m_x, m_y) using the expected grid.
4. Fit the affine grid x = c + m·a1 + n·a2 (least squares). Output pitch, rotation, residuals and pitch uncertainty.
5. For A–B–A data, compute the ratio of DOE pitch to reference pitch (accounting for the reference's odd-order indexing), the drift-corrected mean, and the standard deviation.
6. Convert to angle using λ from the Bristol log and Λ_ref. Report θ, uncertainty and a pass/fail against 0.66° ± 0.001°.
7. Log everything (CSV/JSON) with timestamps; make plots of residuals and Allan deviation.

## Implementation status (2026-11-03)
Bare-bones single-pair analysis is built: `ArgusEye.py` (CLI, report, outputs), `spots.py` (detection, indexing, centroids), `lattice.py` (camera model and fits), `sources.py` (Frame + stub hooks for Arena capture, Bristol λ, T/P/RH, Ciddor), `config.ini`. Run with `.venv\Scripts\python.exe ArgusEye.py`.
- Current parts: reference Holo/Or MS-460 (8 × 6, odd orders, nominal 0.62° separation used until it is characterized), sample Holo/Or MS-500 (6 × 4). The 9 × 5 Neurophos DOE is handled (odd counts include the zero order) but not yet measured.
- Separation angle = angle between neighbouring spots (−1/+1 for even counts, 0/+1 for odd counts); also reports −1↔+1 and the full angle between outermost spots, per axis (2·asin(m·λ/Λ)).
- Camera model (2026-11-12): pinhole f·tanθ + Brown–Conrady distortion, OpenCV convention, single F (fx = fy). Fits F, principal point, rotation, the reference zero order's offset from the lens axis, and the terms in `[detection] distortion` (default `k1`). Chosen over the earlier degree-3 2D polynomial on the 11-frame MS-460 set: k1 alone (7 params) predicts left-out spots at 0.093 px vs 0.100 px for the polynomial (18 params). Calibrated on the inner spots only, it predicts the outer ring at 0.10 px vs 0.24 px for the polynomial; degree 4 and k2 diverge. This matters because the MS-500 (±2.9°) and the 9×5 DOE (±2.6°) reach beyond the reference field (±2.2° in x). Distortion-model uncertainty = shift when p1 p2 are toggled, counted only if that model fits (rms ≤ 1.5× main), /√3. The earlier ±1-degree rule counted an inadequate degree-2 fit and inflated the term to ~1300 ppm.
- Earlier first-image notes: strong off-centre distortion and oversized spots (σ ≈ 13 px vs ~2.5 px expected) in the first images suggest the camera is not at the back focal plane and/or the beam is off-centre on the lens.
- Angles are reported at 1310 nm in air; λ cancels in the substitution, no air-index correction yet. Pass/fail only when a spec is set per sample in config.ini.
- Reference period (2026-11-12): `[reference] period_file` reads KeyMaster's `grating_period.json` (priority over `period_um`, then `separation_deg`). KeyMaster x = ArgusEye x (the reference's 8-spot direction); y = x while `y_measured` is false. `traceable: false` (or the nominal separation) flags every result INTERIM. Current file: Keyence-calibration value 244.244 µm, u = 5025 ppm (k = 1); a traceable stage-micrometer value (~30 ppm, k = 2) will replace it in the same format.
- Images (2026-11-12): Arena Mono16 .raw + .raw.json sidecar (12-bit data in the top bits), in `Images/`, gitignored. Naming `<part>_<Ref|Smp>_<yyyymmdd-hhmm>_<nn|dark>.raw`; config paths accept wildcards. First set: MS-460 reference, 11 frames + 1 dark (`MS-460-2-N-A_Ref_20261112-1642_*`): frame-to-frame spot jitter ≈ 0.008 px; the 0.078 px fit residual is a fixed per-spot pattern (repeatable to 0.003 px), not noise.
- MS-500 result (2026-11-12, `MS-500-1-Y-A_Smp_20261112-1725_*`, 11 frames + dark): −1/+1 = 1.15348° X, 1.15382° Y at 1310 nm (period 130.14 / 130.11 µm), interim (KeyMaster period not traceable). Datasheet: 0.93° at the 1064 nm design wavelength = 1.1450° at 1310 nm, so we measure +0.74% (≈1.5σ of the 5025 ppm reference period).
  - Open issue: the sample fits the reference camera at 0.35 px rms (reference 0.085 px). The residuals are symmetric and radial: the sample sees more barrel than the reference (joint-fit k1 −5.0 vs −4.5). A joint camera fit moves the result +620 ppm X / +310 ppm Y, and this is not in the uncertainty budget yet. Suspected cause: a different axial position of the patterned surface (substrate thickness / pattern side / holder), which changes where each order crosses the lens. Tilt is aligned by iris retro-reflection (~0.1°, i.e. ~1 ppm) and is unlikely to be the cause.
  - Dark level fell 187 → 75 counts between sets with no exposure/gain change, likely sensor temperature (warm-up). Each set has its own dark.
- Next steps (pending user decisions):
  1. Measure the MS-500 on the Keyence via KeyMaster; also the MS-460's y axis if possible. Compare the period ratio MS-500/MS-460, where the Keyence scale cancels. ArgusEye ratio: 0.53284 X / 0.53268 Y with the reference-only camera; the joint fit gives 0.53317 / 0.53285.
  2. Proposed, not yet approved:
     - a `[sample] period_file` comparison;
     - the joint-fit check added to the report and the uncertainty budget;
     - `spec_wavelength_nm`, to scale datasheet specs to 1310 nm;
     - reporting the zero-order leak strength.
  3. Bench work:
     - match the axial position of both gratings, then take two reference shots ~2 mm apart to measure the k1 sensitivity;
     - give the camera a 30–60 min warm-up;
     - log the camera temperature, if Arena exposes DeviceTemperature.
- Deferred: A–B–A sequencing, Allan deviation, live acquisition, dark frames from capture, tilt checks.

## Related Keyence task (later)
Comparator measurement of the reference period: certified scale (Edmund #59-276, 25 µm divisions, NIST certificate) stacked chrome-down over the grating, ~50 µm gap. Step along a 22 mm diameter, locate the grating pattern against the nearest certified tick near the image centre, and fit the period. Correct the soda-lime scale (≈ 9 ppm/°C) to its calibration temperature and report Λ at 20 °C (fused silica 0.55 ppm/°C). Target ≈ 30 ppm (k = 2).
