#!/usr/bin/env python
"""
bookend - paired coordinate-break "bookends" for OpticStudio sequential lenses.

bookend first redefines the fields: it finds the largest field in the file
(radial distance from the axis) and replaces all fields with four points on
the +Y axis at 0%, 50%, 70% and 100% of it, each with weight 1 and no
vignetting factors.  The field type (angle, object height, ...) is kept.
Use --keep-fields to leave the fields alone.

Then, for every refractive component, bookend:

  1. fixes the clear semi-diameter of every lens surface at its current value;
  2. converts every Standard / Even Asphere lens surface to Zernike Standard Sag
     (maximum term 11, normalization radius = the fixed semi-diameter, all
     Zernike coefficients zero, so the sag is unchanged);
  3. wraps every lens surface in its own coordinate-break pair (pivot at the
     surface vertex);
  4. wraps every singlet, and every assembly of elements in contact (cemented,
     or air-spaced with a zero gap), in one outer coordinate-break pair (pivot
     at the component's front vertex).  Elements inside an assembly do not get
     their own pair.

Mirrors are skipped.  Surface types that cannot become Zernike Standard Sag
without changing shape (toroidal, biconic, odd asphere, freeform, ...) keep
their type and are flagged, but still get a fixed semi-diameter and a CB pair.

At nominal (all decenters and tilts zero) the new lens is optically identical to
the original.  bookend checks this (surface vertex positions, surface sag, real
rays, EFL, every configuration) and refuses to save if anything moved.

Layout produced for a singlet (S1, S2):

    C1 front       CB     component decenter/tilt, thickness 0
    C1.s1 front    CB     surface decenter/tilt, thickness 0
    (S1)                  Zernike Standard Sag, thickness moved -> 0
    C1.s1 rear     CB     pickups x(-1), order 1, carries S1's thickness
    C1.s2 front    CB
    (S2)
    C1.s2 rear     CB     thickness 0
    C1 return      dummy  Position solve: back to the C1 front vertex
    C1 rear        CB     pickups x(-1), order 1, Position solve: forward
    C1 gap         dummy  carries S2's air gap (and any solve on it)

The first column is the Comment written on each inserted row; the original
lens surfaces keep their own comments.  The decenter/tilt parameters to
perturb are on the "front" rows.

Usage:
    python bookend.py lens.zos                 # writes lens_bookend.zos
    python bookend.py lens.zmx -o out.zmx
    python bookend.py lens.zos --dry-run       # show the plan, change nothing

Requirements: OpticStudio with a ZOS-API licence (Professional / Premium /
Enterprise), Python 3.8+, pythonnet 3.x (pip install pythonnet).
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass

# Comment written on every inserted row, e.g. "C2 front" or "C2.s1 rear".
TAG_RE = re.compile(r"^(C\d+)(?:\.s(\d+))?\s+(front|rear|return|gap)$")

AIR_NAMES = {"", "AIR", "VACUUM"}
CONVERTIBLE_TYPES = {"Standard", "EvenAspheric"}
ZERNIKE_TYPE = "ZernikeStandardSag"
BLOCKING_TYPES = {"CoordinateBreak", "NonSequentialComponent"}
FIXED_SOLVES = {"Fixed", "None"}

# Thickness solves that keep their meaning when the thickness moves from a lens
# surface onto the row that follows it.  Anything else (edge thickness, ZPL
# macro, ...) depends on the lens surface itself, so it is frozen and flagged.
MOVABLE_SOLVES = {"Variable", "SurfacePickup", "MarginalRayHeight", "ChiefRayHeight",
                  "Position", "OpticalPathDifference", "CenterOfCurvature",
                  "PupilPosition", "Compensator"}

# Merit-function operands that constrain thickness; flagged for review when
# they reference a lens surface (single-surface operands) or span one (range
# operands, Param1..Param2).
SURFACE_THICKNESS_OPERANDS = {"CTGT", "CTLT", "CTVA", "ETGT", "ETLT", "ETVA",
                              "COGT", "COLT", "COVA"}
RANGE_THICKNESS_OPERANDS = {"TTHI", "TTGT", "TTLT", "TTVA", "MNCT", "MXCT", "MNCA",
                            "MXCA", "MNCG", "MXCG", "MNET", "MXET", "MNEA", "MXEA",
                            "MNEG", "MXEG"}

# New field points, as fractions of the largest field, all on the +Y axis.
FIELD_FRACTIONS = (0.0, 0.5, 0.7, 1.0)
# Multi-configuration operands that override field definitions per configuration.
MCE_FIELD_OPERANDS = {"XFIE", "YFIE", "FLWT", "FVDX", "FVDY", "FVCX", "FVCY", "FVAN", "FLTP"}

ASPHERE_HEADER_RE =re.compile(r"\d+\s*(st|nd|rd|th)\s*order|r\s*\^\s*\d+", re.I)
ZERNIKE_COEFF_RE = re.compile(r"^zernike\s*(\d+)", re.I)

# Test rays for the before/after comparison: normalized field and pupil points.
TEST_FIELDS = [(0.0, 0.0), (0.0, 0.7), (0.0, 1.0), (0.7, 0.0), (1.0, 0.0)]
TEST_PUPIL = [(0.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (0.5, 0.5)]
SAG_FRACTIONS = [0.25, 0.5, 0.75, 1.0]


class BookendError(Exception):
    pass


# --------------------------------------------------------------------------
# Lens analysis (plain Python, no OpticStudio needed)
# --------------------------------------------------------------------------

@dataclass
class SurfaceRecord:
    index: int
    type_name: str
    material: str
    thickness: float
    comment: str
    model_glass: bool = False

    @property
    def is_mirror(self):
        return self.material.strip().upper() == "MIRROR"

    @property
    def is_glass(self):
        if self.is_mirror:
            return False
        return self.model_glass or self.material.strip().upper() not in AIR_NAMES


@dataclass
class Unit:
    uid: str
    kind: str        # "singlet" | "cemented assembly" | "contact assembly"
    first: int
    last: int
    n_elements: int

    @property
    def surfaces(self):
        return list(range(self.first, self.last + 1))

    @property
    def new_surface_count(self):
        return 2 * len(self.surfaces) + 4


def find_units(records, contact_tol=1e-9):
    """Group lens surfaces into singlets and contact assemblies.

    records[i] must describe surface i; the last record is the image surface.
    Returns (units, notes).
    """
    notes = []
    image = len(records) - 1
    components = []   # [first, last, n_glass]
    i = 1
    while i < image:
        rec = records[i]
        if rec.is_mirror:
            notes.append(f"Surface {i}: mirror - skipped.")
            i += 1
            continue
        if not rec.is_glass or rec.type_name in BLOCKING_TYPES:
            i += 1
            continue
        start = j = i
        while j < image and records[j].is_glass and records[j].type_name not in BLOCKING_TYPES:
            j += 1
        closing = records[j]
        reason = None
        if j >= image:
            reason = "glass continues to the image surface"
        elif closing.is_mirror:
            reason = f"surface {j} is a mirror (Mangin element) - mirrors are skipped"
        elif closing.type_name in BLOCKING_TYPES:
            reason = f"surface {j} is a {closing.type_name} inside the element"
        if reason:
            notes.append(f"Surfaces {start}-{j}: {reason}; left untouched.")
        else:
            components.append([start, j, j - start])
        i = j + 1

    # Merge components whose air gap is zero (option A: one pair around the stack).
    groups = []
    for comp in components:
        if groups:
            prev = groups[-1][-1]
            if comp[0] == prev[1] + 1 and abs(records[prev[1]].thickness) <= contact_tol:
                groups[-1].append(comp)
                continue
        groups.append([comp])

    units = []
    for n, group in enumerate(groups, 1):
        n_el = sum(c[2] for c in group)
        if len(group) > 1:
            kind = "contact assembly"
        elif n_el > 1:
            kind = "cemented assembly"
        else:
            kind = "singlet"
        units.append(Unit(f"C{n}", kind, group[0][0], group[-1][1], n_el))
    return units, notes


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.lines = []
        self.flags = []

    def info(self, msg=""):
        print(msg)
        self.lines.append(msg)

    def flag(self, msg):
        print("FLAG: " + msg)
        self.lines.append("FLAG: " + msg)
        self.flags.append(msg)

    def write(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


# --------------------------------------------------------------------------
# OpticStudio connection (standard ZOS-API standalone boilerplate)
# --------------------------------------------------------------------------

class OpticStudio:
    def __init__(self, zemax_path=None):
        self.zemax_path = zemax_path
        self.app = None
        self.connection = None

    def __enter__(self):
        import clr
        import winreg

        key = winreg.OpenKey(winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER),
                             r"Software\Zemax", 0, winreg.KEY_READ)
        zemax_root = winreg.QueryValueEx(key, "ZemaxRoot")[0]
        winreg.CloseKey(key)
        clr.AddReference(os.path.join(zemax_root, r"ZOS-API\Libraries\ZOSAPI_NetHelper.dll"))
        import ZOSAPI_NetHelper

        if self.zemax_path:
            ok = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(self.zemax_path)
        else:
            ok = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize()
        if not ok:
            raise BookendError("Unable to locate OpticStudio; try --zemax-path.")
        zdir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        clr.AddReference(os.path.join(zdir, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(zdir, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI

        self.ZOSAPI = ZOSAPI
        self.connection = ZOSAPI.ZOSAPI_Connection()
        if self.connection is None:
            raise BookendError("Unable to initialize the .NET connection to ZOS-API.")
        self.app = self.connection.CreateNewApplication()
        if self.app is None:
            raise BookendError("Unable to start an OpticStudio instance.")
        if not self.app.IsValidLicenseForAPI:
            raise BookendError("This OpticStudio licence is not valid for ZOS-API use.")
        self.system = self.app.PrimarySystem
        if self.system is None:
            raise BookendError("Unable to acquire the primary optical system.")
        return self

    def __exit__(self, *exc):
        if self.app is not None:
            self.app.CloseApplication()
            self.app = None
        self.connection = None
        return False


def solve_name(cell):
    try:
        return str(cell.Solve)
    except Exception:
        return "Unknown"


def close_enough(a, b, tol):
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------
# The work
# --------------------------------------------------------------------------

class Bookend:
    def __init__(self, zos, report, max_term=11):
        self.zos = zos
        self.system = zos.system
        self.report = report
        self.max_term = max_term
        api = zos.ZOSAPI
        self.SurfaceType = api.Editors.LDE.SurfaceType
        self.SurfaceColumn = api.Editors.LDE.SurfaceColumn
        self.SolveType = api.Editors.SolveType
        self.MeritOperandType = api.Editors.MFE.MeritOperandType
        self.MeritColumn = api.Editors.MFE.MeritColumn

    @property
    def lde(self):
        return self.system.LDE

    # ---- file & records --------------------------------------------------

    def load(self, path):
        if not self.system.LoadFile(path, False):
            raise BookendError(f"OpticStudio could not open {path}")
        if str(self.system.Mode) != "Sequential":
            raise BookendError("bookend works on sequential systems only.")

    def read_records(self):
        records = []
        for i in range(self.lde.NumberOfSurfaces):
            row = self.lde.GetSurfaceAt(i)
            records.append(SurfaceRecord(
                index=i,
                type_name=str(row.Type),
                material=str(row.Material or ""),
                thickness=float(row.Thickness),
                comment=str(row.Comment or ""),
                model_glass="Model" in solve_name(row.MaterialCell),
            ))
        return records

    # ---- fields ------------------------------------------------------------

    def field_units(self):
        fields = self.system.SystemData.Fields
        if "Angle" in str(fields.GetFieldType()):
            return "deg"
        return str(self.system.SystemData.Units.LensUnits).lower()

    def current_fields(self):
        fields = self.system.SystemData.Fields
        out = []
        for i in range(1, fields.NumberOfFields + 1):
            f = fields.GetField(i)
            vig = any(abs(float(getattr(f, v))) > 0 for v in ("VDX", "VDY", "VCX", "VCY", "VAN"))
            out.append((float(f.X), float(f.Y), float(f.Weight), vig))
        return out

    def largest_field(self):
        """(radial size, field number) of the largest field."""
        best, number = 0.0, None
        for i, (x, y, _, _) in enumerate(self.current_fields(), 1):
            r = math.hypot(x, y)
            if r > best:
                best, number = r, i
        return best, number

    def redefine_fields(self, fmax):
        """Replace all fields with FIELD_FRACTIONS of fmax on +Y, weight 1, no vignetting."""
        fields = self.system.SystemData.Fields
        while fields.NumberOfFields > 1:
            fields.RemoveField(fields.NumberOfFields)
        first = fields.GetField(1)
        first.SetXFixed()
        first.SetYFixed()
        first.X, first.Y, first.Weight = 0.0, FIELD_FRACTIONS[0] * fmax, 1.0
        for frac in FIELD_FRACTIONS[1:]:
            fields.AddField(0.0, frac * fmax, 1.0)
        fields.ClearVignetting()
        for i in range(1, fields.NumberOfFields + 1):
            f = fields.GetField(i)
            f.VDX = f.VDY = f.VCX = f.VCY = f.VAN = 0.0
            f.Ignore = False

        mce = self.system.MCE
        ops = sorted({str(mce.GetOperandAt(i).Type) for i in range(1, mce.NumberOfOperands + 1)}
                     & MCE_FIELD_OPERANDS)
        if ops:
            self.report.flag("The Multi-Configuration Editor has field operands (" + ", ".join(ops) +
                             ") that override the new fields in each configuration; please review them.")
        if self.system.MFE.NumberOfOperands > 0:
            self.report.flag("The merit function was built for the old fields. If it is a default "
                             "merit function, regenerate it so it uses the new fields.")

    # ---- cell helpers ----------------------------------------------------

    def par(self, n):
        return getattr(self.SurfaceColumn, f"Par{n}")

    def param_cells(self, row):
        """(header, cell) for every active parameter cell of a surface."""
        out = []
        for n in range(0, 255):
            col = getattr(self.SurfaceColumn, f"Par{n}", None)
            if col is None:
                break
            try:
                cell = row.GetSurfaceCell(col)
                if cell is None or not cell.IsActive:
                    continue
                out.append((str(cell.Header or "").strip().lower(), cell))
            except Exception:
                continue
        return out

    @staticmethod
    def set_int(cell, value):
        try:
            cell.IntegerValue = int(value)
        except Exception:
            cell.DoubleValue = float(value)

    @staticmethod
    def get_int(cell):
        try:
            return int(cell.IntegerValue)
        except Exception:
            return int(round(cell.DoubleValue))

    def make_fixed(self, cell):
        try:
            cell.MakeSolveFixed()
        except Exception:
            cell.SetSolveData(cell.CreateSolveType(self.SolveType.Fixed))

    def set_pickup(self, cell, from_surface, column, scale):
        solve = cell.CreateSolveType(self.SolveType.SurfacePickup)
        solve._S_SurfacePickup.Surface = from_surface
        solve._S_SurfacePickup.ScaleFactor = scale
        solve._S_SurfacePickup.Offset = 0.0
        solve._S_SurfacePickup.Column = column
        cell.SetSolveData(solve)

    def set_position(self, cell, from_surface, length=0.0):
        solve = cell.CreateSolveType(self.SolveType.Position)
        solve._S_Position.FromSurface = from_surface
        solve._S_Position.Length = length
        cell.SetSolveData(solve)

    # ---- phase A: semi-diameter + Zernike --------------------------------

    def prepare_surface(self, idx, label):
        row = self.lde.GetSurfaceAt(idx)
        sd = float(row.SemiDiameter)
        self.make_fixed(row.SemiDiameterCell)
        row.SemiDiameter = sd

        type_name = str(row.Type)
        if type_name == ZERNIKE_TYPE:
            self._update_existing_zernike(row, label, sd)
        elif type_name in CONVERTIBLE_TYPES:
            self._convert_to_zernike(row, label, sd)
        else:
            self.report.flag(
                f"{label}: surface type '{row.TypeName}' cannot be converted to Zernike "
                f"Standard Sag without changing its shape. Left as {row.TypeName} "
                f"(semi-diameter fixed, CB pair still added).")

    def _zernike_cells(self, row):
        cells = self.param_cells(row)
        max_cell = next((c for h, c in cells if "max" in h and "term" in h), None)
        norm_cell = next((c for h, c in cells if "norm" in h and "rad" in h), None)
        coeffs = {}
        for h, c in cells:
            m = ZERNIKE_COEFF_RE.match(h)
            if m:
                coeffs[int(m.group(1))] = float(c.DoubleValue)
        return max_cell, norm_cell, coeffs

    def _convert_to_zernike(self, row, label, sd):
        original_type = row.Type
        saved = []
        if str(original_type) != "Standard":
            for n in range(1, 9):
                cell = row.GetSurfaceCell(self.par(n))
                saved.append((n, float(cell.DoubleValue), solve_name(cell), cell.GetSolveData()))

        settings = row.GetSurfaceTypeSettings(self.SurfaceType.ZernikeStandardSag)
        if not row.ChangeType(settings) or str(row.Type) != ZERNIKE_TYPE:
            self.report.flag(f"{label}: OpticStudio refused the change to Zernike Standard Sag; left as-is.")
            return

        max_cell, norm_cell, _ = self._zernike_cells(row)
        if not self._restore_params(row, saved) or max_cell is None or norm_cell is None:
            self._revert_type(row, original_type, saved)
            self.report.flag(
                f"{label}: could not carry the {row.TypeName} terms into Zernike Standard "
                f"Sag unchanged; reverted to {str(original_type)}.")
            return

        self.set_int(max_cell, self.max_term)
        if sd > 0:
            norm_cell.DoubleValue = sd
            self.report.info(f"  {label}: {str(original_type)} -> Zernike Standard Sag "
                             f"(max term {self.max_term}, norm radius {sd:.6g})")
        else:
            self.report.flag(f"{label}: semi-diameter is {sd}; normalization radius left at its default.")

    def _restore_params(self, row, saved):
        for n, value, kind, solve in saved:
            cell = row.GetSurfaceCell(self.par(n))
            if not close_enough(float(cell.DoubleValue), value, 1e-14):
                # Only write into the column if it really is an aspheric term.
                if not ASPHERE_HEADER_RE.search(str(cell.Header or "")):
                    return False
                cell.DoubleValue = value
            now = solve_name(cell)
            if kind == "Variable" and now != "Variable":
                cell.MakeSolveVariable()
            elif kind not in FIXED_SOLVES | {"Variable"} and now != kind:
                try:
                    cell.SetSolveData(solve)
                except Exception:
                    return False
        return True

    def _revert_type(self, row, original_type, saved):
        row.ChangeType(row.GetSurfaceTypeSettings(original_type))
        for n, value, kind, solve in saved:
            cell = row.GetSurfaceCell(self.par(n))
            cell.DoubleValue = value
            if kind == "Variable":
                cell.MakeSolveVariable()
            elif kind not in FIXED_SOLVES:
                try:
                    cell.SetSolveData(solve)
                except Exception:
                    pass

    def _update_existing_zernike(self, row, label, sd):
        max_cell, norm_cell, coeffs = self._zernike_cells(row)
        if max_cell is None or norm_cell is None:
            self.report.flag(f"{label}: already Zernike Standard Sag but its columns could not be read; left as-is.")
            return
        nonzero = {t: v for t, v in coeffs.items() if v != 0.0}
        beyond = sorted(t for t in nonzero if t > self.max_term)
        if beyond:
            self.report.flag(f"{label}: already Zernike Standard Sag with non-zero terms "
                             f"{beyond} above {self.max_term}; maximum term left at {self.get_int(max_cell)}.")
        else:
            self.set_int(max_cell, self.max_term)
        if nonzero:
            if not close_enough(float(norm_cell.DoubleValue), sd, 1e-12):
                self.report.flag(f"{label}: has non-zero Zernike terms, so its normalization radius "
                                 f"({norm_cell.DoubleValue:.6g}) was kept instead of the semi-diameter ({sd:.6g}).")
        elif sd > 0:
            norm_cell.DoubleValue = sd
        self.report.info(f"  {label}: already Zernike Standard Sag - updated")

    # ---- phase B: coordinate breaks ----------------------------------------

    def insert(self, index, tag, cb):
        row = self.lde.InsertNewSurfaceAt(index)
        if cb:
            row.ChangeType(row.GetSurfaceTypeSettings(self.SurfaceType.CoordinateBreak))
        else:
            try:
                row.DrawData.DoNotDrawThisSurface = True
            except Exception:
                pass
        row.Comment = tag
        return row

    def configure_rear_cb(self, rear_idx, front_idx):
        rear = self.lde.GetSurfaceAt(rear_idx)
        for n in range(1, 6):  # decenter X, decenter Y, tilt X, tilt Y, tilt Z
            self.set_pickup(rear.GetSurfaceCell(self.par(n)), front_idx, self.par(n), -1.0)
        self.set_int(rear.GetSurfaceCell(self.par(6)), 1)  # order: tilt, then decenter

    def build_unit(self, unit):
        """Insert all bookends for one unit.  Units must be built last-to-first."""
        f, n, uid = unit.first, len(unit.surfaces), unit.uid
        self.insert(f, f"{uid} front", cb=True)
        for k in range(1, n + 1):
            base = f + 1 + 3 * (k - 1)
            self.insert(base, f"{uid}.s{k} front", cb=True)
            self.insert(base + 2, f"{uid}.s{k} rear", cb=True)
        end = f + 1 + 3 * n
        self.insert(end, f"{uid} return", cb=False)
        self.insert(end + 1, f"{uid} rear", cb=True)
        self.insert(end + 2, f"{uid} gap", cb=False)

        lens = [f + 2 + 3 * (k - 1) for k in range(1, n + 1)]
        carriers = [f + 3 + 3 * (k - 1) for k in range(1, n)] + [end + 2]
        mapping = dict(zip(lens, carriers))

        self.retarget_thickness_references(mapping)
        for k, (li, ci) in enumerate(mapping.items(), 1):
            self.move_thickness(li, ci, f"{uid}.s{k}")

        for k in range(1, n + 1):
            self.configure_rear_cb(f + 3 + 3 * (k - 1), f + 1 + 3 * (k - 1))

        medium = str(self.lde.GetSurfaceAt(lens[-1]).Material or "")
        ret, gap = self.lde.GetSurfaceAt(end), self.lde.GetSurfaceAt(end + 2)
        if medium:
            ret.Material = medium
            gap.Material = medium
        self.set_position(ret.ThicknessCell, f, 0.0)
        self.configure_rear_cb(end + 1, f)
        self.set_position(self.lde.GetSurfaceAt(end + 1).ThicknessCell, end, 0.0)

    def move_thickness(self, src_idx, dst_idx, label):
        src, dst = self.lde.GetSurfaceAt(src_idx), self.lde.GetSurfaceAt(dst_idx)
        cell = src.ThicknessCell
        kind = solve_name(cell)
        dst.Thickness = float(src.Thickness)
        if kind == "Variable":
            dst.ThicknessCell.MakeSolveVariable()
        elif kind in MOVABLE_SOLVES:
            try:
                dst.ThicknessCell.SetSolveData(cell.GetSolveData())
                moved = solve_name(dst.ThicknessCell) == kind
            except Exception:
                moved = False
            if not moved:
                self.report.flag(f"{label}: could not move the {kind} thickness solve; "
                                 f"frozen at {dst.Thickness:.6g}.")
        elif kind not in FIXED_SOLVES:
            self.report.flag(f"{label}: {kind} thickness solve depends on the lens surface itself "
                             f"and cannot move to the coordinate break; frozen at {dst.Thickness:.6g}.")
        self.make_fixed(cell)
        src.Thickness = 0.0

    def retarget_thickness_references(self, mapping):
        """Point thickness pickups, MCE THIC and TDE TTHI at the new thickness carriers."""
        for i in range(self.lde.NumberOfSurfaces):
            cell = self.lde.GetSurfaceAt(i).ThicknessCell
            if solve_name(cell) != "SurfacePickup":
                continue
            solve = cell.GetSolveData()
            pickup = solve._S_SurfacePickup
            source = int(pickup.Surface)
            if source not in mapping:
                continue
            column = str(pickup.Column)
            if not pickup.IsPickupFromCurrentColumn() and "Thickness" not in column:
                continue
            pickup.Surface = mapping[source]
            cell.SetSolveData(solve)

        mce = self.system.MCE
        for i in range(1, mce.NumberOfOperands + 1):
            op = mce.GetOperandAt(i)
            if str(op.Type) == "THIC" and int(op.Param1) in mapping:
                op.Param1 = mapping[int(op.Param1)]

        try:
            tde = self.system.TDE
            for i in range(1, tde.NumberOfOperands + 1):
                op = tde.GetOperandAt(i)
                if str(op.Type) != "TTHI":
                    continue
                if int(op.Param1) in mapping:
                    op.Param1 = mapping[int(op.Param1)]
                if int(op.Param2) in mapping:
                    op.Param2 = mapping[int(op.Param2)]
        except Exception as exc:
            self.report.flag(f"Could not update TTHI tolerance operands ({exc}); please review the TDE.")

    # ---- layout bookkeeping -------------------------------------------------

    def existing_tags(self):
        return [i for i in range(self.lde.NumberOfSurfaces)
                if TAG_RE.match(str(self.lde.GetSurfaceAt(i).Comment or "").strip()) is not None]

    def original_index_map(self):
        """Original surface number -> current surface number (untagged rows in order)."""
        mapping, orig = {}, 0
        for i in range(self.lde.NumberOfSurfaces):
            if not TAG_RE.match(str(self.lde.GetSurfaceAt(i).Comment or "").strip()):
                mapping[orig] = i
                orig += 1
        return mapping

    def flag_merit_operands(self, lens_surfaces):
        mfe = self.system.MFE
        hits = []
        for i in range(1, mfe.NumberOfOperands + 1):
            op = mfe.GetOperandAt(i)
            name = str(op.Type)
            if name not in SURFACE_THICKNESS_OPERANDS | RANGE_THICKNESS_OPERANDS:
                continue
            try:
                p1 = int(op.GetOperandCell(self.MeritColumn.Param1).IntegerValue)
                p2 = int(op.GetOperandCell(self.MeritColumn.Param2).IntegerValue)
            except Exception:
                hits.append(f"row {i} {name}(?)")
                continue
            if name in SURFACE_THICKNESS_OPERANDS:
                hit = p1 in lens_surfaces
            else:
                lo, hi = min(p1, p2), max(p1, p2)
                hit = any(lo <= s <= hi for s in lens_surfaces)
            if hit:
                hits.append(f"row {i} {name}({p1},{p2})")
        if hits:
            self.report.flag("Merit-function thickness operands reference lens surfaces whose thickness "
                             "now lives on the rear coordinate break; please review: " + ", ".join(hits))

    # ---- verification ------------------------------------------------------

    def primary_wave(self):
        waves = self.system.SystemData.Wavelengths
        for i in range(1, waves.NumberOfWavelengths + 1):
            if waves.GetWavelength(i).IsPrimary:
                return i
        return 1

    def sag_points(self, lens_surfaces):
        points = {}
        for s in lens_surfaces:
            sd = float(self.lde.GetSurfaceAt(s).SemiDiameter)
            points[s] = [(fx * sd, fy * sd) for f in SAG_FRACTIONS
                         for fx, fy in ((0.0, f), (f, 0.0), (f * 0.7071, f * 0.7071))]
        return points

    def snapshot(self, index_map, sag_points):
        """Vertex frames, sags, real rays and EFL for every configuration."""
        mce, mfe = self.system.MCE, self.system.MFE
        wave = self.primary_wave()
        current = mce.CurrentConfiguration
        data = {}
        for config in range(1, mce.NumberOfConfigurations + 1):
            mce.SetCurrentConfiguration(config)
            d = {}
            for orig, cur in index_map.items():
                if orig == 0:
                    continue
                res = self.lde.GetGlobalMatrix(cur, *([0.0] * 12))
                for name, value in zip(("R11", "R12", "R13", "R21", "R22", "R23",
                                        "R31", "R32", "R33", "X", "Y", "Z"), res[1:]):
                    d[f"surface {orig} {name}"] = float(value)
            for orig, pts in sag_points.items():
                for x, y in pts:
                    res = self.lde.GetSag(index_map[orig], x, y, 0.0, 0.0)
                    d[f"surface {orig} sag({x:.4g},{y:.4g})"] = float(res[1]) if res[0] else float("nan")
            image = self.lde.NumberOfSurfaces - 1
            d["EFFL"] = float(mfe.GetOperandValue(self.MeritOperandType.EFFL, 0, wave, 0, 0, 0, 0, 0, 0))
            for hx, hy in TEST_FIELDS:
                for px, py in TEST_PUPIL:
                    for op in ("REAX", "REAY"):
                        value = mfe.GetOperandValue(getattr(self.MeritOperandType, op),
                                                    image, wave, hx, hy, px, py, 0, 0)
                        d[f"{op} H({hx},{hy}) P({px},{py})"] = float(value)
            data[config] = d
        mce.SetCurrentConfiguration(current)
        data["MF"] = float(mfe.CalculateMeritFunction())
        return data


def compare(before, after, tol):
    problems = []
    for config, values in before.items():
        if config == "MF":
            continue
        for key, a in values.items():
            b = after[config].get(key)
            if not close_enough(a, b, tol):
                problems.append(f"config {config}: {key}: {a:.12g} -> {b if b is None else format(b, '.12g')}")
    return problems


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def print_plan(report, records, units, notes):
    report.info(f"{'Unit':<6}{'Kind':<20}{'Surfaces':<12}{'Elements':<10}New rows")
    for u in units:
        report.info(f"{u.uid:<6}{u.kind:<20}{f'{u.first}-{u.last}':<12}{u.n_elements:<10}{u.new_surface_count}")
    report.info(f"Total new rows: {sum(u.new_surface_count for u in units)}")
    for note in notes:
        report.flag(note)
    for u in units:
        for s in u.surfaces:
            t = records[s].type_name
            if t not in CONVERTIBLE_TYPES and t != ZERNIKE_TYPE:
                report.flag(f"Surface {s} ({u.uid}): type {t} will not be converted to Zernike Standard Sag.")


def run(args):
    report = Report()
    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        raise BookendError(f"File not found: {src}")
    stem, ext = os.path.splitext(src)
    out = os.path.abspath(args.output) if args.output else f"{stem}_bookend{ext}"
    if os.path.normcase(out) == os.path.normcase(src):
        raise BookendError("Output must differ from the input file.")

    with OpticStudio(args.zemax_path) as zos:
        bk = Bookend(zos, report, max_term=args.max_term)
        bk.load(src)
        if bk.existing_tags():
            raise BookendError(f"{src} already contains bookend rows (comments like 'C1 front'); bookend has been run on it before.")

        records = bk.read_records()
        units, notes = find_units(records, args.contact_tol)
        report.info(f"bookend: {src}")
        report.info("")
        print_plan(report, records, units, notes)

        fmax = None
        if not args.keep_fields:
            fmax, number = bk.largest_field()
            units_label = bk.field_units()
            report.info("")
            report.info(f"Fields ({str(zos.system.SystemData.Fields.GetFieldType())}, {units_label}):")
            old = bk.current_fields()
            report.info("  current: " + ", ".join(f"({x:.6g}, {y:.6g}) w{w:g}" for x, y, w, _ in old))
            if fmax <= 0:
                report.flag("The largest field is 0 (on-axis only); fields left unchanged.")
                fmax = None
            else:
                report.info(f"  largest: {fmax:.6g} {units_label} (field {number})")
                report.info("  new:     " + ", ".join(f"(0, {f * fmax:.6g})" for f in FIELD_FRACTIONS)
                            + ", all weight 1, no vignetting factors")
                if any(vig for *_, vig in old):
                    report.info("  (existing vignetting factors will be cleared)")

        if args.dry_run or not units:
            if not units:
                report.info("No refractive components found; nothing to do.")
            return 0

        if fmax is not None:
            bk.redefine_fields(fmax)

        lens_surfaces = [s for u in units for s in u.surfaces]
        identity = {i: i for i in range(len(records))}
        sag_points = bk.sag_points(lens_surfaces)
        baseline = bk.snapshot(identity, sag_points)

        report.info("")
        report.info("Fixing semi-diameters and converting surfaces:")
        for u in units:
            for k, s in enumerate(u.surfaces, 1):
                bk.prepare_surface(s, f"Surface {s} ({u.uid}.s{k})")
        problems = compare(baseline, bk.snapshot(identity, sag_points), args.tol)

        report.info("")
        report.info("Inserting coordinate breaks...")
        for u in reversed(units):
            bk.build_unit(u)
        index_map = bk.original_index_map()
        if len(index_map) != len(records):
            raise BookendError("Surface bookkeeping failed: original surfaces could not be identified.")
        final = bk.snapshot(index_map, sag_points)
        problems += [p for p in compare(baseline, final, args.tol) if p not in problems]
        bk.flag_merit_operands({index_map[s] for s in lens_surfaces})

        report.info("")
        report.info(f"Verification ({len(baseline) - 1} configuration(s)):")
        if problems:
            report.info(f"  FAILED - {len(problems)} value(s) changed, first ones:")
            for p in problems[:15]:
                report.info("    " + p)
        else:
            report.info("  vertex positions, surface sags, real rays and EFL unchanged")
        if not close_enough(baseline["MF"], final["MF"], args.tol):
            report.flag(f"Merit function changed {baseline['MF']:.10g} -> {final['MF']:.10g}; "
                        f"check operands that reference surface numbers or thicknesses.")

        report_path = os.path.splitext(out)[0] + "_report.txt"
        if problems and not args.force:
            report.info("")
            report.info("Not saved (use --force to save anyway).")
            report.write(report_path)
            return 2
        bk.system.SaveAs(out)
        report.info("")
        report.info(f"Saved {out}")
        report.info(f"{len(report.flags)} flag(s); report written to {report_path}")
        report.write(report_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bookend",
        description="Wrap every lens surface, singlet and contact assembly of a sequential "
                    "OpticStudio file in paired coordinate breaks, and convert lens surfaces "
                    "to Zernike Standard Sag.")
    parser.add_argument("input", help="OpticStudio sequential file (.zos or .zmx)")
    parser.add_argument("-o", "--output", help="output file (default: <input>_bookend.<ext>)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without changing anything")
    parser.add_argument("--keep-fields", action="store_true",
                        help="leave the field definitions unchanged")
    parser.add_argument("--max-term",type=int, default=11, help="Zernike Standard Sag maximum term (default 11)")
    parser.add_argument("--contact-tol", type=float, default=1e-9,
                        help="air gap treated as contact, in lens units (default 1e-9)")
    parser.add_argument("--tol", type=float, default=1e-8, help="relative tolerance for verification (default 1e-8)")
    parser.add_argument("--force", action="store_true", help="save even if verification finds differences")
    parser.add_argument("--zemax-path", help="OpticStudio install folder, if not the registered one")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except BookendError as exc:
        print(f"bookend: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
