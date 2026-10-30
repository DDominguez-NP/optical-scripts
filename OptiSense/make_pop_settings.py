# make_pop_settings: writes the POP settings file that the pixel-readout macro (ZPL21) reads,
# <lens name>_POPtext.CFG next to the lens file, so the macro can be used in the OpticStudio GUI.
# OptiSense writes this file itself at the start of every run that calls a ZPLM macro.
#
# Usage: python make_pop_settings.py "C:\path\to\lens.zmx"
# Rerun it whenever the lens's POP settings change.

import os
import sys

from PythonStandaloneApplication import PythonStandaloneApplication
from OptiSense import write_pop_text_settings


def main():
    if len(sys.argv) != 2:
        print('Usage: python make_pop_settings.py "C:\\path\\to\\lens.zmx"')
        return 1
    lens_file = os.path.normpath(sys.argv[1])
    if not os.path.exists(lens_file):
        print(f"[Error] Lens file not found: {lens_file}")
        return 1

    zos = PythonStandaloneApplication()
    try:
        if not zos.TheSystem.LoadFile(lens_file, False):
            print(f"[Error] OpticStudio could not load the lens file: {lens_file}")
            return 1
        settings_path = write_pop_text_settings(zos.TheSystem, zos.ZOSAPI, lens_file)
        print(f"[*] POP settings for the macro saved to {settings_path}")
    finally:
        zos.TheApplication.CloseApplication()
        zos.TheApplication = None
    return 0


if __name__ == '__main__':
    sys.exit(main())
