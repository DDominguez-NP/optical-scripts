# Keyence VHX metadata from saved TIFFs.
#
# The VHX-X1 writes a private "KmsFile" maker note into the EXIF of every TIFF. Entries used
# here (found by comparing files, not from Keyence documentation):
#   17    calibration value; um/px of the saved image = value / TAG17_FACTOR
#   286   zoom (e.g. 100, 500)
#   1555  stage X, 1556 stage Y, 1557 stage Z (um)
#   2     capture time;  72  controller model (e.g. "VHX-X1")
# TAG17_FACTOR was measured on 2026-11-09 at x100: the on-screen measuring tool, the saved
# pixels and entry 17 agree to 1 ppm before and after a recalibration. It has not been
# checked independently at other zooms (x500 agrees with x100 to 0.18%).

import struct
import datetime

import tifffile

TAG17_FACTOR = 2.37319


def _blocks(path):
    with tifffile.TiffFile(path) as t:
        exif = t.pages[0].tags.get('ExifTag')
        if exif is None:
            return None
        mn = exif.value.get('MakerNote') if isinstance(exif.value, dict) else None
    if not mn or not mn.startswith(b'KmsFile'):
        return None
    n = struct.unpack_from('<H', mn, 8)[0]
    entries = {}
    with open(path, 'rb') as f:
        for i in range(n):
            tag, typ, cnt = struct.unpack_from('<HHI', mn, 10 + 12 * i)
            raw = mn[10 + 12 * i + 8: 10 + 12 * i + 12]
            if typ == 7 and cnt > 8:
                f.seek(struct.unpack('<I', raw)[0])
                entries[tag] = f.read(cnt)
    return entries


def _value(block):
    if block is None or len(block) < 6:
        return None
    code = struct.unpack_from('<H', block, 4)[0]
    p = block[6:]
    if code in (5, 7) and len(p) == 8:
        v = struct.unpack('<d', p)[0]
        if code == 7:
            return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=v)
        return v
    if code == 8 and len(p) >= 4:
        nbytes = struct.unpack_from('<I', p, 0)[0]
        return p[4:4 + nbytes].decode('utf-16-le', 'replace').strip('\x00')
    return None


def read_meta(path, factor=TAG17_FACTOR):
    """Calibration, zoom, stage position and time from a VHX TIFF, or None if absent."""
    try:
        b = _blocks(path)
    except Exception:
        return None
    if not b or 17 not in b:
        return None
    cal = _value(b.get(17))
    if not isinstance(cal, float) or cal <= 0:
        return None
    t = _value(b.get(2))
    return dict(
        calibration_value=cal,
        um_per_px=cal / factor,
        zoom=_value(b.get(286)),
        stage_x_um=_value(b.get(1555)),
        stage_y_um=_value(b.get(1556)),
        stage_z_um=_value(b.get(1557)),
        captured=t.isoformat(timespec='seconds') if isinstance(t, datetime.datetime) else None,
        model=_value(b.get(72)),
    )
