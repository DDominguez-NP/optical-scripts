"""
POP pixel readout: the % of beam power falling in each pixel of a 3x3 block centred on a
target pixel, and the target pixel's crosstalk in dB.

Python version of the ZPL21 macro (macros/ZPL21.ZPL), for scripts that drive OpticStudio
through the ZOS-API. Same physics, same numbers, same return slots; much faster, because
POP's irradiance grid comes back as an array instead of a text file parsed in ZPL.

Usable from any script with an open OpticStudio system:

    settings = PixelSettings.from_zpl(r"...\\Macros\\ZPL21.ZPL")
    result = measure_pixel_block(system, zosapi, config=5, target=(-0.9576, -1.9153),
                                 pop_settings_file=r"...\\lens_POPtext.CFG", settings=settings)
    result.crosstalk_db, result.target_pct, result.block, result.slot(n)

The binning itself (bin_3x3) is pure numpy and needs no OpticStudio.
"""

import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass

import numpy as np

# Return slots, numbered as in ZPL21: 0 = target pixel %, 1..9 = 3x3 block, 10 = crosstalk dB
SLOT_TARGET = 0
SLOT_DB = 10
FAILED = -1.0              # ZPL21's value for every slot when the chief ray fails
CROSSTALK_FLOOR_DB = -100.0


@dataclass(frozen=True)
class PixelSettings:
    """Pixel pitch (lens units), the POP field and wavelength numbers, and the beam-centre
    sample offset (centre sample = n/2 + c_off, 1-based), as in ZPL21's user settings."""
    pitch: float
    pop_field: int = 1
    pop_wave: int = 1
    c_off: int = 1

    @classmethod
    def from_zpl(cls, macro_path):
        """Reads the settings from the user-settings block of a ZPL21 macro file, so the
        macro stays the one place to change them."""
        with open(macro_path, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', 'replace')
        values = {}
        for name in ('pitch', 'pop_field', 'pop_wave', 'c_off'):
            m = re.search(rf'^\s*{name}\s*=\s*([-+0-9.eE]+)', text, re.MULTILINE)
            if not m:
                raise ValueError(f"'{name} = ...' not found in {macro_path}")
            values[name] = float(m.group(1))
        return cls(pitch=values['pitch'], pop_field=int(values['pop_field']),
                   pop_wave=int(values['pop_wave']), c_off=int(values['c_off']))


@dataclass(frozen=True)
class PixelResult:
    """block[a, b]: % of total beam power in pixel (a, b) of the 3x3 block, a = x index
    (left to right), b = y index (bottom to top); block[1, 1] is the target pixel.
    ok is False when the chief ray failed (every slot then reads -1, as in ZPL21)."""
    block: np.ndarray
    ok: bool = True
    grid_covers_block: bool = True
    chief_ray: tuple = (math.nan, math.nan)
    spacing: tuple = (math.nan, math.nan)

    @property
    def target_pct(self):
        return self.slot(SLOT_TARGET)

    @property
    def crosstalk_db(self):
        return self.slot(SLOT_DB)

    def slot(self, n):
        """The value ZPL21 returns in OPTRETURN slot n."""
        if not self.ok:
            return FAILED
        if n == SLOT_TARGET:
            return float(self.block[1, 1])
        if 1 <= n <= 9:
            return float(self.block[(n - 1) // 3, (n - 1) % 3])
        if n == SLOT_DB:
            outside = 1 - self.block[1, 1] / 100
            return float(10 * math.log10(outside)) if outside > 1e-10 else CROSSTALK_FLOOR_DB
        raise ValueError(f"no slot {n} (slots are 0-10)")


def bin_3x3(irradiance, dx, dy, chief_ray, target, pitch, c_off=1):
    """
    Bins a POP irradiance grid into the 3x3 block of pixels centred on target.

    irradiance[row, col]: rows bottom to top (y), columns left to right (x), as POP lists
    them. The grid is centred on the chief ray: 1-based sample n/2 + c_off sits at
    chief_ray. Every sample counts toward the total; each sample is assigned to the pixel
    containing its centre. Returns (block as % of total power, grid_covers_block).
    """
    irradiance = np.asarray(irradiance, dtype=float)
    ny, nx = irradiance.shape
    x_cr, y_cr = chief_ray
    x_t, y_t = target
    i0 = nx // 2 + c_off - 1          # 0-based index of the chief-ray sample
    j0 = ny // 2 + c_off - 1
    x = x_cr + (np.arange(nx) - i0) * dx
    y = y_cr + (np.arange(ny) - j0) * dy

    half = 1.5 * pitch
    covers = bool(x[0] <= x_t - half and x[-1] >= x_t + half and
                  y[0] <= y_t - half and y[-1] >= y_t + half)

    def pixel_index(coord, centre):
        u = (coord - centre) / pitch
        k = np.floor(u + 1.5).astype(int)
        k[(u < -1.5) | (u >= 1.5)] = -1
        return k

    kx = pixel_index(x, x_t)
    ky = pixel_index(y, y_t)
    block = np.zeros((3, 3))
    for a in range(3):
        cols = kx == a
        for b in range(3):
            block[a, b] = irradiance[np.ix_(ky == b, cols)].sum()
    total = irradiance.sum()
    if total > 0:
        block *= 100 / total
    return block, covers


def _net_array_to_numpy(values):
    """Copies a .NET double[,] into a numpy array (fast path via a pinned buffer)."""
    rows, cols = values.GetLength(0), values.GetLength(1)
    try:
        import ctypes
        from System.Runtime.InteropServices import GCHandle, GCHandleType
        handle = GCHandle.Alloc(values, GCHandleType.Pinned)
        try:
            address = handle.AddrOfPinnedObject().ToInt64()
            buffer = (ctypes.c_double * (rows * cols)).from_address(address)
            return np.ctypeslib.as_array(buffer).reshape(rows, cols).copy()
        finally:
            handle.Free()
    except Exception:
        return np.array([[values[r, c] for c in range(cols)] for r in range(rows)])


def chief_ray_landing(system, zosapi, settings):
    """
    Traces the chief ray of the POP field and wavelength to the image surface, as ZPL21
    does (normalized field assumes radial field normalization). Returns (x, y) in the
    image surface's local coordinates, or None if the ray fails.
    """
    fields = system.SystemData.Fields
    field = fields.GetField(settings.pop_field)
    max_field = max(math.hypot(fields.GetField(i).X, fields.GetField(i).Y)
                    for i in range(1, fields.NumberOfFields + 1))
    hx = field.X / max_field if max_field > 0 else 0.0
    hy = field.Y / max_field if max_field > 0 else 0.0
    image_surface = system.LDE.NumberOfSurfaces - 1

    trace = system.Tools.OpenBatchRayTrace()
    try:
        rays = trace.CreateNormUnpol(1, zosapi.Tools.RayTrace.RaysType.Real, image_surface)
        no_opd = getattr(zosapi.Tools.RayTrace.OPDMode, 'None')   # 'None' is a Python keyword
        rays.AddRay(settings.pop_wave, hx, hy, 0.0, 0.0, no_opd)
        trace.RunAndWaitForCompletion()
        rays.StartReadingResults()
        # out parameters: ray number, error code, vignette code, X, Y, Z, L, M, N, l2, m2, n2, OPD, intensity
        out = rays.ReadNextResult(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    finally:
        trace.Close()
    success, error_code, x, y = out[0], out[2], out[4], out[5]
    if not success or error_code != 0:
        return None
    return x, y


def run_pop_irradiance(system, zosapi, pop_settings_file):
    """
    Runs POP from a copy of the given settings file (linear irradiance, no saved beam,
    no zoom) and returns (irradiance[row, col], dx, dy). Rows run bottom to top, as in
    POP's text listing. The settings file itself is only read.
    """
    if not os.path.isfile(pop_settings_file):
        raise FileNotFoundError(f"POP settings file not found: {pop_settings_file}")
    copy = os.path.join(tempfile.gettempdir(), f"pop_crosstalk_{uuid.uuid4().hex}.CFG")
    shutil.copyfile(pop_settings_file, copy)
    pop = system.Analyses.New_Analysis_SettingsFirst(
        zosapi.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    try:
        s = pop.GetSettings().__implementation__
        if not s.LoadFrom(copy):
            raise RuntimeError(f"OpticStudio could not load POP settings from {pop_settings_file}")
        # a log scale would return log10 values clipped at the scale's floor; the text
        # listing ZPL21 reads is always linear
        s.LogScale = zosapi.Analysis.PhysicalOptics.POPScaleTypes.Linear
        s.SaveOutputBeam = False
        s.ZoomIn = zosapi.Analysis.PhysicalOptics.POPZoomTypes.NoZoom
        pop.ApplyAndWaitForCompletion()
        grid = pop.GetResults().GetDataGrid(0)
        if grid is None:
            raise RuntimeError("POP returned no data (check the POP settings and the lens)")
        return _net_array_to_numpy(grid.Values), grid.Dx, grid.Dy
    finally:
        pop.Close()
        try:
            os.remove(copy)
        except OSError:
            pass


def measure_pixel_block(system, zosapi, config, target, pop_settings_file, settings):
    """
    Measures the 3x3 pixel block around target = (x, y) (image-surface local coordinates,
    lens units) in the given configuration. Equivalent to a ZPL21 call with Hy = x,
    Px = y in that configuration. Leaves the system in that configuration.
    """
    system.MCE.SetCurrentConfiguration(int(config))
    chief = chief_ray_landing(system, zosapi, settings)
    if chief is None:
        return PixelResult(block=np.full((3, 3), FAILED), ok=False)
    irradiance, dx, dy = run_pop_irradiance(system, zosapi, pop_settings_file)
    block, covers = bin_3x3(irradiance, dx, dy, chief, target, settings.pitch, settings.c_off)
    return PixelResult(block=block, grid_covers_block=covers, chief_ray=chief, spacing=(dx, dy))
