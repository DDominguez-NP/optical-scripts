# Stage-micrometer profiles and the calibration certificate.
#
# A profile describes the physical layout of one scale: division, which ticks are longer
# (the hierarchy used to number ticks), tick style, usable range and zones to keep out of the
# fit (centre cross, square outlines). Values marked layout_assumed have not been checked
# against the real part; confirm them on a Keyence image of the scale and override them in
# config.ini [scale] if they differ.
#
# The certificate gives "as measured" positions at certified points. Tick positions between
# certified points use linear interpolation of the deviation (measured - aim).

import csv
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class ScaleProfile:
    name: str
    model: str
    division_um: float
    # (every n divisions, rank): a tick whose index is a multiple of n is at least that rank.
    # Rank 1 = minor tick; higher ranks are longer ticks.
    tick_levels: list
    tick_style: str                 # 'symmetric' (ticks cross the baseline) or 'one-sided'
    line_width_um: float
    range_mm: tuple                 # usable scale coordinates
    exclude_mm: list = field(default_factory=list)   # [(lo, hi)] scale coordinates to ignore
    label_every: int = 40
    layout_assumed: bool = True
    notes: str = ""

    def level(self, k):
        """Expected tick rank for integer tick indices k."""
        k = np.asarray(k, dtype=np.int64)
        lv = np.ones(k.shape, dtype=int)
        for every, rank in self.tick_levels:
            lv = np.where(k % every == 0, np.maximum(lv, rank), lv)
        return lv

    def excluded(self, x_um):
        x_mm = np.asarray(x_um) * 1e-3
        bad = (x_mm < self.range_mm[0]) | (x_mm > self.range_mm[1])
        for lo, hi in self.exclude_mm:
            bad |= (x_mm >= lo) & (x_mm <= hi)
        return bad


PROFILES = {
    "edmund59276_metric": ScaleProfile(
        name="edmund59276_metric", model="Edmund #59-276, metric cross (X axis)",
        division_um=25.0, tick_levels=[(4, 2), (20, 3), (40, 4)], tick_style="symmetric",
        line_width_um=3.0, range_mm=(-12.0, 12.0), exclude_mm=[(-0.15, 0.15)], label_every=40,
        notes="0 at the metric cross. Tick hierarchy, line width and range are assumed; "
              "add the square outlines to exclude_mm once their positions are known."),
    "edmund59276_english": ScaleProfile(
        name="edmund59276_english", model="Edmund #59-276, English cross (X axis)",
        division_um=25.4, tick_levels=[(5, 2), (10, 3), (50, 4)], tick_style="symmetric",
        line_width_um=3.0, range_mm=(-11.43, 11.43), exclude_mm=[(-0.15, 0.15)],
        label_every=50,
        notes="1 mil divisions, 0 at the English cross. Layout assumed; the NIST certificate "
              "must cover this axis before it is used for a traceable result."),
    "sm4": ScaleProfile(
        name="sm4", model="Applied Image SM-4 (25 mm X axis)",
        division_um=10.0, tick_levels=[(5, 2), (10, 3), (100, 4)], tick_style="one-sided",
        line_width_um=2.5, range_mm=(0.0, 25.0), exclude_mm=[], label_every=100,
        notes="0..25 mm. Tick hierarchy, line width and one-sided style are assumed."),
}


def _parse_ranges(text):
    out = []
    for part in text.replace(';', ',').split(','):
        part = part.strip()
        if part:
            lo, hi = part.split(':')
            out.append((float(lo), float(hi)))
    return out


def load_profile(section):
    """Profile named in config [scale] profile, with any layout overrides from that section."""
    name = section.get('profile', '').strip()
    if name not in PROFILES:
        raise ValueError(f"[scale] profile must be one of {', '.join(PROFILES)} (got '{name}')")
    p = PROFILES[name]
    changes = {}
    if section.get('division_um', '').strip():
        changes['division_um'] = float(section['division_um'])
    if section.get('tick_levels', '').strip():
        levels = []
        for part in section['tick_levels'].split(','):
            every, rank = part.split(':')
            levels.append((int(every), int(rank)))
        changes['tick_levels'] = levels
    if section.get('tick_style', '').strip():
        changes['tick_style'] = section['tick_style'].strip()
    if section.get('line_width_um', '').strip():
        changes['line_width_um'] = float(section['line_width_um'])
    if section.get('range_mm', '').strip():
        lo, hi = (float(v) for v in section['range_mm'].split(','))
        changes['range_mm'] = (lo, hi)
    if section.get('exclude_mm', None) is not None and section.get('exclude_mm').strip():
        changes['exclude_mm'] = _parse_ranges(section['exclude_mm'])
    if changes:
        p = replace(p, **changes)
    return p


class Certificate:
    """Certified positions; deviation (measured - aim) interpolated linearly between points."""

    def __init__(self, aim_um, measured_um, U_um=None, U_default_um=0.0, correlation=0.0,
                 source="", certified=True):
        order = np.argsort(aim_um)
        self.aim = np.asarray(aim_um, float)[order]
        self.dev = (np.asarray(measured_um, float) - np.asarray(aim_um, float))[order]
        self.U = None if U_um is None else np.asarray(U_um, float)[order]
        self.U_default = float(U_default_um)
        self.correlation = float(correlation)
        self.source = source
        self.certified = certified

    @classmethod
    def from_csv(cls, path, units="mm", U_default_um=0.0, correlation=0.0):
        scale = {"mm": 1000.0, "um": 1.0, "inch": 25400.0, "in": 25400.0}[units]
        aim, meas, U = [], [], []
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                row = {k.strip().lower(): (v or '').strip() for k, v in row.items() if k}
                if not row.get('aim_' + units) and not row.get('aim'):
                    continue
                a = row.get('aim_' + units) or row.get('aim')
                m = row.get('measured_' + units) or row.get('measured')
                aim.append(float(a) * scale)
                meas.append(float(m) * scale)
                U.append(float(row['u_um']) if row.get('u_um') else np.nan)
        if len(aim) < 2:
            raise ValueError(f"certificate {path} needs at least two certified points")
        U = np.array(U)
        return cls(aim, meas, None if np.all(np.isnan(U)) else np.where(np.isnan(U), U_default_um, U),
                   U_default_um, correlation, source=str(path))

    @classmethod
    def nominal(cls, U_default_um, correlation=0.0):
        """No certificate: nominal positions. Results are flagged as not traceable."""
        return cls([-1e9, 1e9], [-1e9, 1e9], None, U_default_um, correlation,
                   source="none (nominal positions)", certified=False)

    def deviation(self, x_um):
        return np.interp(x_um, self.aim, self.dev)

    def position(self, x_nominal_um):
        """True (certified) position of a mark at nominal position x."""
        return np.asarray(x_nominal_um, float) + self.deviation(x_nominal_um)

    def covers(self, x_um):
        x = np.asarray(x_um)
        return (x >= self.aim[0] - 1e-6) & (x <= self.aim[-1] + 1e-6)

    def U_at(self, x_um):
        if self.U is None:
            return np.full(np.shape(x_um), self.U_default)
        return np.interp(x_um, self.aim, self.U)

    def interpolation_u_um(self):
        """Leave-one-out RMS of the linear deviation interpolation (needs >= 3 points)."""
        if not self.certified or len(self.aim) < 3:
            return 0.0
        res = []
        for i in range(1, len(self.aim) - 1):
            keep = np.ones(len(self.aim), bool)
            keep[i] = False
            res.append(np.interp(self.aim[i], self.aim[keep], self.dev[keep]) - self.dev[i])
        return float(np.sqrt(np.mean(np.square(res))))
