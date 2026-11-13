# Grating period measurement with a Keyence microscope and a certified stage micrometer: script specification

Context handoff for Claude Code. Owner: Daniel Dominguez, Neurophos. 2026-11-05.

## 1. Purpose

Measure the average period Λ of a 1D diffraction grating (the Argus reference grating, Holo/Or, nominal Λ ≈ 242.12 µm, Ø25.4 mm part, Ø22.9 mm clear aperture, fused silica) to about **30–50 ppm (k = 2)**. A Keyence digital microscope images the grating through a certified glass stage micrometer laid over it. This is an interim calibration until an external Littrow calibration is available. It must be far inside the DOE tolerance (±1500 ppm).

The script turns a set of microscope images plus metadata into Λ at 20 °C, its uncertainty, uniformity residuals and QC plots.

## 2. Physical setup (what the images contain)

- **Stack:** the grating sits face up in an aluminium fixture. The certified scale lies **chrome side down** over it, about 50 µm above the grating surface (25–100 µm, set by shims). The microscope looks down through the scale glass. Each image shows the scale ticks and the grating pattern together.
- **Scale:**
  - **Edmund #59-276**, dual-axis linear scale with NIST certificate: chrome on glass, 25.4 × 76.2 × 1.50 mm, metric scale with 25 µm divisions, numbered major marks.
  - Alternative: **Applied Image SM-4** (25 mm X axis, 10 µm divisions, soda-lime 1.52 mm; ANAB certificate gives "As Measured" positions at 0.01–1 mm steps and 2, 4, 6, 8, 10, 14, 18, 20, 22, 25 mm; expanded uncertainty 0.415 µm).
  - Use the **metric X scale**. Keep the Y scale and the centre cross off the measurement path.
- **Orientation:** the grating lines are parallel to the scale ticks, so the grating period direction runs along the scale axis. Any leftover angle α is measured and corrected.
- **Path:** the grating centre is placed over the **11 mm** tick, and the measurement runs along a diameter through the grating centre, ~0 → 22 mm. Both ends are anchored on certified lines (0 and 22 mm, or 2 and 22 mm).
- **Stepping:** the stage steps ≈ 10 periods (≈ 2.42 mm) at a time; ~9–10 stops per pass. At each stop, a grating feature and its nearest tick are brought near the image centre. Passes: forward + reverse, ×3, plus one pass with the grating rotated 180°.
- **Imaging:**
  - Fixed magnification, focus, lighting and exposure throughout. Lossless 8- or 16-bit TIFF/PNG, no enhancement, auto-contrast, sharpening or depth composition.
  - Target pixel size ≈ 0.5–1 µm. Earlier Keyence images at ~1.04 µm/px were 2880 × 2160 px; the field should hold ≥ 2 grating periods and ≥ 20 ticks.
- **Environment:** log the temperature at the scale at the start and end of each pass (±0.1 °C ideal, ±0.5 °C acceptable).

## 3. Inputs

### 3.1 Images

One file per stop. Naming: `pass{P}_stop{S}.tif`.

### 3.2 Run metadata CSV (`run.csv`)

| column | meaning |
|---|---|
| `file` | image filename |
| `pass`, `stop` | integers |
| `direction` | `fwd` / `rev` |
| `grating_rot_deg` | 0 or 180 |
| `stage_x_mm` | approximate stage X from the Keyence readout (±0.05 mm is enough). Used only to resolve tick and period indices; never used as a length. |
| `stage_y_mm` | optional |
| `temp_c` | temperature at capture (interpolate between start/end if needed) |
| `notes` | optional |

### 3.3 Config (`config.yaml`)

```yaml
grating:
  nominal_period_um: 242.12
  cte_ppm_per_c: 0.55        # fused silica
  orientation: 1D
scale:
  model: "Edmund 59-276"     # or "Applied Image SM-4"
  division_um: 25.0          # 10.0 for SM-4
  cte_ppm_per_c: 9.0         # soda-lime; confirm
  cert_temp_c: 20.0          # from certificate (convert from °F if needed)
  cert_file: cert.csv
  cert_U_um_k2: 0.415        # per-position expanded uncertainty (k=2); replace with the Edmund value
  cert_correlation: 0.0      # 0 = independent end errors (conservative), 1 = fully correlated
imaging:
  pixel_um_nominal: 0.5      # starting guess only; measured per image from ticks
  scale_band: auto           # or [y0, y1] px rows containing ticks
  grating_band: auto         # or [y0, y1] px rows used for the grating profile
  center_window_px: 200      # use features within this distance of the image centre
analysis:
  reference_temp_c: 20.0
  outlier_sigma: 3.0
```

### 3.4 Certificate table (`cert.csv`)

| column | meaning |
|---|---|
| `aim_mm` | nominal position |
| `measured_mm` | certified "as measured" position |
| `U_um` | optional, per-point expanded uncertainty |

Ticks between certified points use linear interpolation of the *deviation* (measured − aim).

## 4. Algorithm

### 4.1 Per image

1. **Load and normalise.** Convert to float and flat-field if a blank image is provided. Record a saturation fraction.
2. **Find the tick direction.** Estimate the tick-line angle with a Hough transform or a structure tensor. Rotate the analysis coordinates so the ticks are vertical; keep everything in sub-pixel coordinates without resampling the image if possible.
3. **Locate the bands.** Find the scale band (rows with a strong periodic signal at ≈ division / pixel) and the grating band (rows with a strong signal at ≈ Λ / pixel). `auto` uses the row-wise FFT power at the two expected frequencies.
4. **Fit the tick centres.**
   - Average the scale band over the tick length to get a 1D profile. Chrome lines may be bright or dark depending on the lighting; auto-detect polarity.
   - Find the peaks, then refine each centre sub-pixel (Gaussian or parabolic fit to the line profile, or a symmetric-centroid method).
   - Target ≤ 0.05 px repeatability.
5. **Identify the ticks.**
   - Detect major ticks (longer lines or numbers) from the line lengths in the band.
   - Combine with `stage_x_mm` to assign an absolute division index *k* to each tick: position_nominal = k × division.
   - Fail loudly if the assignment is ambiguous.
6. **Local scale.**
   - Fit the tick pixel position against k × division (linear, optionally quadratic, to see distortion) to get µm/px and the offset.
   - Report the residual RMS (QC: should be ≤ 0.05 µm for a good scale and image).
7. **Grating position.**
   - Average the grating band along the line direction to get a 1D profile.
   - Measure the pattern position with one of two methods:
     - **(a) Phase method (default):** project the profile onto the fundamental exp(−i2πx/P), with P = Λ/µm-per-px refined from the profile FFT. The phase gives the position of a defined pattern feature modulo P.
     - **(b) Template method:** cross-correlate with a one-period template from the first image (sub-pixel peak by parabolic interpolation).
   - Choose the feature instance nearest the image centre (within `center_window_px`).
   - Also measure the grating line angle; α = grating angle − tick angle.
8. **Convert to scale coordinates.**
   - Take the tick k* nearest to the grating feature.
   - X_feature = X_cert(k*) + (x_feature_px − x_tick_px(k*)) × (µm/px)_local, where X_cert includes the certificate deviation.
   - The offset term is at most half a division (≤ 12.5 µm), so magnification errors are negligible.
   - Record the feature's distance from the image centre (for the height-parallax check).

### 4.2 Across images

9. **Assign period numbers.**
   - n = round((X_feature − X_first)/Λ_nominal).
   - Check that |X_feature − X_first − n·Λ_nominal| < Λ/4. If not, flag it: a period skip or a wrong tick ID.
10. **Fit.**
    - Weighted least squares X = A + Λ_axis · n, separately for each pass and for all passes pooled (common slope, separate intercepts per pass).
    - Optional regressor: offset of the feature from the image centre (absorbs height-parallax differences between the scale and grating planes).
    - Reject points with residuals > `outlier_sigma` and report them.
11. **Corrections.**
    - Angle: Λ_T = Λ_axis · cos α (use the mean α; it should be < 5 mrad).
    - Scale temperature: Λ_T ← Λ_T × [1 + α_s (T_run − T_cert)].
    - Report at 20 °C: Λ_20 = Λ_T / [1 + α_g (T_run − 20)].
12. **Central value.** Also fit only the stops within the central ±1.5 mm (central Ø3 mm) as a uniformity check, with its own (larger) uncertainty.

## 5. Uncertainty budget (report each term in µm and ppm, k = 1 and k = 2)

| term | model | typical (k = 2) over 20–22 mm |
|---|---|---|
| Scale certificate | end-to-end length: U = U_pt·√(2(1−ρ)) … U_pt (ρ = correlation) | 21–29 ppm with U_pt = 0.415 µm |
| Tick and grating location | from per-image residual scatter, propagated through the fit | ≈ 10–15 ppm |
| Temperature | (α_s − α_g)·u(T) plus α_g term | ≈ 4–9 ppm for ±0.5 °C |
| Angle | Λ(1 − cos α) uncertainty | < 1 ppm for α < 2 mrad |
| Height parallax | slope of the residual vs centre-offset regressor | ≈ few ppm |
| Repeatability | spread of the per-pass slopes / √N | measured |
| Interpolation between certified ticks | deviation-interpolation residual | small if end points are certified |

Combine as a root sum of squares. Report Λ_20 ± U (k = 2). Pass/fail versus a configurable target (default 50 ppm).

## 6. Outputs

- `results.json`: Λ_20, U (k = 2), each budget term, per-pass slopes, α, temperatures, version and config hash.
- `stops.csv`: per image: tick fit µm/px, residual RMS, k*, feature px, X_feature, n, residual, QC flags.
- Plots (PNG):
  - X vs n with fit;
  - residuals vs position (uniformity);
  - per-pass slopes;
  - µm/px vs stop (stability);
  - tick-fit residuals;
  - an annotated example image (detected ticks, grating feature, centre window).
- A short text summary for the lab notebook.

## 7. Validation (build first)

1. **Synthetic image generator.**
   - Inputs: known pixel size (with small per-image variation and a quadratic distortion term), tick pattern with major ticks, binary grating profile of period Λ_true at a small angle, noise, blur, ±50 µm height offset (a slightly different magnification for the grating layer), and per-stop stage jitter of ±50 µm.
   - The script must recover Λ_true to < 2 ppm with the noise off and stay within its stated uncertainty with the noise on.
2. **Single real stop.** Check the tick-fit residuals and grating-phase stability over 10 repeated captures without moving anything (target ≤ 0.05 µm).
3. **Reversal test.** The forward/reverse and 0°/180° passes must agree within the uncertainty.

## 8. Secondary mode: single field (no stepping)

For quick checks: one image of the scale and one image of the grating, at the same height and magnification and in the same field region, with no refocus between them. The scale image gives µm/px; the grating image gives Λ from the profile FFT over ~10 periods. Expected uncertainty ≈ 100–150 ppm (k = 2); report it clearly as such.

## 9. Implementation notes

- Python 3.11. numpy, scipy, scikit-image, tifffile, pandas, matplotlib, pyyaml. CLI: `python measure_period.py --config config.yaml --run run.csv --images ./imgs --out ./out`.
- Keep the core math in pure functions: `fit_ticks`, `grating_phase`, `to_scale_coords`, `assign_periods`, `fit_period`, `uncertainty`, so each can be unit-tested against the synthetic generator.
- Never use the Keyence's built-in µm/px or stage readout as a length. They only seed searches and index assignment.
- Log the software version and a hash of the inputs for traceability.

## 10. Reference numbers

- λ (Argus) = 1310 nm. Reference first-order angle ≈ 0.31°; DOE 9 × 5 spots at 0.66° (period ≈ 113.7 µm).
- Soda-lime CTE ≈ 9 ppm/°C (confirm with the scale vendor). Fused silica 0.55 ppm/°C.
- Applied Image SM-4 certificate: expanded uncertainty 0.415 µm per position (ANAB AC-2818). The Edmund #59-276 certificate values are still to be confirmed.

## 11. Implementation status (2026-11-05)

Plan-of-record method (no longer interim). Reference grating is 2D: Holo/Or MS-460-1310-Y-A, 6 × 8 spots at 0.62°, odd orders only, Λ ≈ 242.12 µm on both axes. Two scales supported: Edmund #59-276 (metric or English cross, mount centres the part on the chosen cross) and Applied Image SM-4 (part centred at 12.5 mm). Both are used chrome side down, so numerals read mirrored.

Built (run with `.venv\Scripts\python.exe KeyMaster.py`; settings in `config.ini`):
- `KeyMaster.py` CLI, per-stop processing (parallel per pass), per-axis analysis, outputs, quick mode (`--quick SCALE [GRATING] --scale-mm X`).
- `scales.py` scale profiles (division, long-tick hierarchy, tick style, range, exclusion zones) and certificate (linear interpolation of the deviation).
- `ticks.py` polarity, background, baseline and angle, tick band, sub-pixel tick centres, tick lengths, absolute numbering from the hierarchy (level-consistency score), cubic fit of certified position vs pixel.
- `grating.py` FFT lattice search, zoomed-DFT peak refinement, unit-cell template folded from the first image of each pass, Fourier coarse match with half-cell ambiguity ratios, real-space Gauss-Newton fit of origin and lattice (masked pixels simply excluded).
- `period.py` stop-to-stop counting on the effective lattice (true, half, or centred for odd-orders-only images), pooled fit with per-pass intercepts, per-pass slopes, consistency z-tests, uncertainty budget.
- `plots.py`, `synth.py` (synthetic runs with known truth), `validate.py` (end-to-end scenarios).

Departures from sections 3–9 above:
- `config.ini` (configparser, like ArgusEye) instead of `config.yaml`; no pandas/pyyaml/scikit-image.
- `run.csv` takes `scale_mm` (approximate scale reading at the image centre, read from the numerals); `stage_x_mm` is used to fill it in within a pass when only one row has it.
- 2D lattice instead of a 1D line profile. For a 2D lattice the same lattice point is tracked, so Λ = slope / cos α (the "· cos α" in §4.2.11 applies to 1D line crossings, which is what `type = 1d` uses).
- Gap parallax is self-calibrated: the in-image period (scale-plane µm/px) is Λ/(1+ε), the stepped fit gives Λ, and each point is corrected by ε·d before the final fit; the regressor (§4.2.10) remains as a residual check when the offsets spread enough.
- Temperature is corrected per stop (scale expansion about its zero, grating about its centre), not once with the run mean, so drift within a pass does not tilt fwd vs rev.
- Ticks are re-read after dividing the fitted grating pattern out of the tick band.
- Statistical term = larger of the pooled-fit SE and the between-pass spread (not both); scale-fit model and gap-correction terms are sensitivity estimates.

## 12. Status (2026-11-13): paused until the stage micrometer arrives

Findings from real Keyence VHX-X1 images (no micrometer yet):
- Odd-orders-only gratings (MS-460-2-N-A 8 × 6, MS-500 10 × 10) image as thin outlines that look the same for either step direction, so the image repeats at Λ/2 on both axes (half-shift scores 0.99–1.00). KeyMaster detects this ("half period on both axes") and reports the full Λ.
- `keyence.py` reads the VHX maker note: µm/px = entry 17 / 2.37319 (checked to 1 ppm against the on-screen tool at ×100, before and after a recalibration); 4K images are scaled by 2880/4000 (checked to 45 ppm). Stage XYZ are entries 1555–1557.
- Magnification changes by about +44 ppm per µm of focus height at ×500.
- MS-460-2-N-A, Keyence calibration only: 244.24 µm ± 1% (k = 2), 0.9% above the 242.12 µm datasheet value (itself only good to about ±0.8% because 0.62° is rounded). Interim period file given to ArgusEye with `traceable: false`.
- MS-500: Keyence ×500 gives 130.459 µm (x), 130.419 µm (y). The ratio to the MS-460 at the same zoom predicts 130.57 µm (1.1497° at 1310 nm) in ArgusEye's units; ArgusEye measured 130.14 µm (1.1535°), −0.33%. Unresolved. Candidates: Keyence zoom repeatability between sessions (test: MS-460 / MS-500 / MS-460 at ×500 without touching the zoom) and ArgusEye extrapolating its camera model beyond the MS-460 spot field (test: fit only spots inside it, vary the distortion degree).

When the micrometer arrives: image it at ×100 to confirm the tick layout and fix `scales.py`/`config.ini`; load its certificate; run the stepped measurement of the MS-460-2-N-A (x, then y at 90°), then the MS-500 the same way; replace both period files with `traceable: true` versions.

Open items: confirm the real tick layouts (hierarchy, line width, numbering, Edmund square outlines → `exclude_mm`) on Keyence images of each scale; load the real certificates (and check that the Edmund NIST certificate covers the axis used); confirm the soda-lime CTE; first real-data checks per §7.2–7.3.
