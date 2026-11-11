# OptiSense: automated tolerance sensitivity analysis for Zemax OpticStudio via the ZOS-API.
#
# Given a lens file (.zmx) and a merit function (.mf), this script:
#   1. Loads the lens and merit function in OpticStudio and records the nominal operand values.
#   2. Writes a tolerance script (OptiSenseScript.tsc) that reports every merit function operand
#      as a separate tolerancing criterion.
#   3. Runs a sensitivity tolerance analysis using that script and saves the results to a .ZTD file.
#   4. Reads the .ZTD file, computes the change in each operand for every tolerance operand, and
#      summarizes each column by sum of absolute values, root sum square, and Raentsch sum
#      (Zeiss approach: sqrt(abs_sum * rss)).
#   5. Exports the nominal values, summaries, and per-tolerance changes to a CSV file.
#
# If the merit function calls a ZPLM macro, OptiSense first writes the POP settings file that the
# POP pixel-readout macro (ZPL21) reads: <lens name>_POPtext.CFG next to the lens file.
#
# Crosstalk (ZPLM 21 rows) runs in two stages unless CrosstalkMode = zpl:
#   A. Tolerancing runs with the ZPLM 21 rows blanked out, on all configured cores, and saves
#      every perturbed (and compensated) lens through SAVE rows OptiSense adds in memory.
#   B. Python measures the ZPLM 21 rows on each saved lens (pop_crosstalk.py, same numbers as
#      the macro, far faster), then re-saves the lens with the full merit function and its own
#      POP settings file so it can be reopened in OpticStudio. The lenses go in
#      <output folder>\<lens name>_TolLenses.
# Before stage A, the nominal crosstalk is computed both ways and the run stops if the macro and
# Python disagree.
#
# File paths and options are read from config.ini, located next to this script.

import os
import sys
# import time
import math
import re
import glob
import shutil
import tempfile
import configparser
import traceback
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

# Ensure PythonStandaloneApplication is in the same directory or Python PATH
from PythonStandaloneApplication import PythonStandaloneApplication
from pop_crosstalk import PixelSettings, measure_pixel_block

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_COLUMNS = ['Operand', 'Comment', 'Min Tol', 'Max Tol']
CANCEL_CHECK_S = 1
DEFAULT_CORES = 8

# ZPLM macro values that mean the macro failed: 0 (macro error) and -1 (chief ray failed)
MACRO_FAILURE_VALUES = (0.0, -1.0)
FAILED_TEXT = 'FAILED'
# POP settings for the pixel-readout macro: the lens's POP settings with no beam file and no zoom
POP_TEXT_SETTINGS_SUFFIX = '_POPtext.CFG'
# PROGRESS_INTERVAL_S = 3
# ESTIMATE_STEP_MIN = 15

# Tolerance Data Editor rows that don't perturb the lens, so they add no merit function evaluations
COMPENSATOR_TYPES = {'COMP', 'CPAR', 'CMCO', 'CEDV'}
NON_TOLERANCE_TYPES = COMPENSATOR_TYPES | {'NULL', 'TOFF', 'COMM', 'SAVE', 'SEED', 'STAT', 'TWAV'}

# Crosstalk macro measured in Python (pop_crosstalk.py) in two-stage runs
CROSSTALK_MACRO = 21
CROSSTALK_MODES = ('python', 'zpl')
# Largest macro-vs-Python difference allowed in the nominal check (% or dB)
CROSSTALK_CHECK_TOLERANCE = 1e-3
# Lens files written by SAVE rows: TSAV_MIN_0001.zmx, TSAV_MAX_0001.zmx, ...
TOL_LENS_GLOB = 'TSAV_M??_????.zmx'
TOL_LENS_FOLDER_SUFFIX = '_TolLenses'

# One ZPLM 21 row of the merit function: its row number, configuration, return slot and target
CrosstalkRow = namedtuple('CrosstalkRow', 'index config slot target')


def load_configuration(config_file=os.path.join(SCRIPT_DIR, "config.ini")):
    """
    Loads the configuration file. If it doesn't exist, generates a template
    and halts the script so the user can fill it out.
    """
    config = configparser.ConfigParser()

    if not os.path.exists(config_file):
        config['PATHS'] = {
            'ZemaxFile': r'C:\Path\To\Your\File.zmx',
            'MeritFunctionFile': r'C:\Path\To\Your\File.mf',
            'OutputDirectory': r'C:\Path\To\Output\Folder'
        }
        config['SETTINGS'] = {
            'InjectAdditionalCommands': 'False',
            'AdditionalCommandsFile': r'C:\Path\To\XtraScript.txt',
            'Cores': str(DEFAULT_CORES),
            'ParallelMacros': 'False',
            'CrosstalkMode': 'python',
            'CrosstalkWorkers': '1'
        }
        with open(config_file, 'w') as f:
            config.write(f)

        print(f"[*] A default configuration file was created at: {config_file}")
        print("[*] Please open it, update your file paths, and run this script again.")
        sys.exit(0)

    config.read(config_file)
    return config


def _cell_text(cell, zosapi):
    """
    Returns a compact string for an MFE cell (integers as-is, doubles without trailing zeros).
    """
    cell_types = zosapi.Editors.CellDataType
    if cell.DataType == cell_types.Integer:
        return str(cell.IntegerValue)
    if cell.DataType == cell_types.Double:
        # Unused parameter cells report inf; show them as 0
        return f"{cell.DoubleValue:g}" if math.isfinite(cell.DoubleValue) else '0'
    return str(cell.Value) or '0'


def load_merit_function(system, zosapi, mf_file):
    """
    Loads and calculates the merit function. Returns the nominal operand values and a
    label for each operand (type followed by its first four parameters), taken directly
    from the Merit Function Editor so labels and values always line up.
    """
    system.MFE.LoadMeritFunction(mf_file)
    # eval_start = time.perf_counter()
    system.MFE.CalculateMeritFunction()
    # eval_seconds = time.perf_counter() - eval_start

    columns = zosapi.Editors.MFE.MeritColumn
    param_columns = [columns.Param1, columns.Param2, columns.Param3, columns.Param4]

    nom_mf_results = []
    operand_labels = []
    for k in range(1, (system.MFE.NumberOfOperands + 1)):
        mf_operand = system.MFE.GetOperandAt(k)

        params = [_cell_text(mf_operand.GetOperandCell(col), zosapi) for col in param_columns]
        operand_labels.append(' '.join([str(mf_operand.TypeName)] + params))

        val = mf_operand.Value
        if val == float('inf') or val == float('-inf') or math.isnan(val):
            nom_mf_results.append(0.0)
        else:
            nom_mf_results.append(float(val))

    # return nom_mf_results, operand_labels, eval_seconds
    return nom_mf_results, operand_labels


def merit_function_calls_macros(mf_file):
    """
    True if the merit function file has a ZPLM row. Read from the file itself so it can be
    checked before the merit function is evaluated (evaluating it already runs the macros).
    """
    with open(mf_file, 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')
    return any(line.startswith('ZPLM') for line in text.splitlines())


def write_pop_text_settings(system, zosapi, lens_file):
    """
    Writes the POP settings file the pixel-readout macro reads: the lens's own POP settings with
    "Save Output Beam" off, so parallel evaluations never share a beam file, and no display zoom,
    so the text listing holds POP's full native grid. Returns the settings file path.
    """
    settings_path = os.path.splitext(lens_file)[0] + POP_TEXT_SETTINGS_SUFFIX
    # SettingsFirst opens the analysis without running POP, which would write the default beam file
    pop = system.Analyses.New_Analysis_SettingsFirst(zosapi.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    try:
        settings = pop.GetSettings().__implementation__
        settings.SaveOutputBeam = False
        settings.ZoomIn = zosapi.Analysis.PhysicalOptics.POPZoomTypes.NoZoom
        if not settings.SaveTo(settings_path):
            raise RuntimeError(f"Could not write the POP settings file: {settings_path}")
    finally:
        pop.Close()
    return settings_path


# ---- Two-stage crosstalk: stage A tolerancing without the macro, stage B in Python -------------

def read_macro_rows(system, zosapi):
    """
    Returns (crosstalk_rows, other_macros) for the merit function in the editor: a CrosstalkRow
    for each ZPLM 21 row (its configuration from the CONF row above it, its target pixel from
    Hy and Px, as ZPL21 reads them), and the numbers of any other macros the rows call.

    A ZPLM row directly below another ZPLM row with the same macro doesn't run the macro again
    in OpticStudio: it reads another slot of the row above's call. So such a row takes the
    target of the row above, whatever its own Hy and Px say (e.g. "ZPLM 21 10 0 0").
    """
    mfe = system.MFE
    columns = zosapi.Editors.MFE.MeritColumn
    types = zosapi.Editors.MFE.MeritOperandType
    config, rows, other = 1, [], set()
    previous = None    # the CrosstalkRow on the row above, if it was one
    for k in range(1, mfe.NumberOfOperands + 1):
        op = mfe.GetOperandAt(k)
        row = None
        if op.Type == types.CONF:
            config = op.GetOperandCell(columns.Param1).IntegerValue
        elif op.Type == types.ZPLM:
            macro = op.GetOperandCell(columns.Param1).IntegerValue
            if macro != CROSSTALK_MACRO:
                other.add(macro)
            else:
                slot = op.GetOperandCell(columns.Param2).IntegerValue
                if not 0 <= slot <= 10:
                    raise ValueError(f"Merit function row {k}: ZPLM {CROSSTALK_MACRO} has no return slot {slot} (slots are 0-10)")
                if previous is not None:
                    target = previous.target
                else:
                    target = (op.GetOperandCell(columns.Param4).DoubleValue, op.GetOperandCell(columns.Param5).DoubleValue)
                row = CrosstalkRow(k, config, slot, target)
                rows.append(row)
        previous = row
    return rows, sorted(other)


def measure_crosstalk_rows(system, zosapi, rows, pop_settings_file, pixel_settings):
    """
    Measures every ZPLM 21 row on the lens in the system, running POP once per configuration
    and target. Returns {merit function row number: value}, as the macro would return it.
    Leaves the system in the configuration it was in.
    """
    start_config = system.MCE.CurrentConfiguration
    results, values = {}, {}
    try:
        for row in rows:
            key = (row.config, row.target)
            if key not in results:
                results[key] = measure_pixel_block(system, zosapi, row.config, row.target,
                                                   pop_settings_file, pixel_settings)
            values[row.index] = results[key].slot(row.slot)
    finally:
        system.MCE.SetCurrentConfiguration(start_config)
    if not all(r.grid_covers_block for r in results.values()):
        print("[Warning] The POP grid does not cover the full 3x3 pixel block for this lens; "
              "the block slots (1-9) are short, the target pixel is unaffected.")
    return values


def check_crosstalk_against_macro(nominal_values, python_values, operand_labels):
    """
    Compares the macro's nominal ZPLM 21 values with Python's. Raises if any differ by more
    than CROSSTALK_CHECK_TOLERANCE, since the Python results would then not match the macro.
    """
    worst = 0.0
    for index, value in python_values.items():
        difference = abs(value - nominal_values[index - 1])
        worst = max(worst, difference)
        if difference > CROSSTALK_CHECK_TOLERANCE:
            raise RuntimeError(
                f"Nominal check failed: '{operand_labels[index - 1]}' is {nominal_values[index - 1]:.6f} from "
                f"the macro but {value:.6f} from Python. Check that pop_crosstalk.py matches the macro, "
                "or set CrosstalkMode = zpl in config.ini to run with the macro.")
    print(f"[*] Nominal crosstalk check passed: macro and Python agree (largest difference {worst:.1e}).")


def write_stage_a_merit_function(system, zosapi, rows, path):
    """
    Saves the merit function in the editor with its ZPLM 21 rows changed to BLNK, so stage A
    never runs the macro while every other row keeps its number.
    """
    mfe = system.MFE
    blank = zosapi.Editors.MFE.MeritOperandType.BLNK
    for row in rows:
        op = mfe.GetOperandAt(row.index)
        op.ChangeType(blank)
    mfe.SaveMeritFunction(path)


def add_save_rows(system, zosapi):
    """
    Replaces any SAVE rows in the Tolerance Data Editor (in memory only) with one after every
    tolerance, numbered 1, 2, ... in order, so tolerance k's lenses are TSAV_MIN/MAX_000k.zmx.
    Returns [(operand type, comment)] for the tolerances in order.
    """
    tde = system.TDE
    tol_types = zosapi.Editors.TDE.ToleranceOperandType
    removed = 0
    for k in range(tde.NumberOfOperands, 0, -1):
        if str(tde.GetOperandAt(k).Type) == 'SAVE':
            tde.RemoveOperandAt(k)
            removed += 1
    if removed:
        print(f"[*] Replaced the lens's {removed} SAVE rows (in memory only) with OptiSense's own.")

    tolerances = []
    k = 1
    while k <= tde.NumberOfOperands:
        op = tde.GetOperandAt(k)
        if str(op.Type) not in NON_TOLERANCE_TYPES:
            tolerances.append((str(op.Type), str(op.Comment)))
            save = tde.InsertNewOperandAt(k + 1)
            save.ChangeType(tol_types.SAVE)
            save.GetOperandCell(zosapi.Editors.TDE.ToleranceColumn.Param1).IntegerValue = len(tolerances)
            k += 1
        k += 1
    return tolerances


def tolerance_lens_path(folder, number, limit):
    """Path of the lens SAVE row number saves at the tolerance's 'MIN' or 'MAX' limit."""
    return os.path.join(folder, f"TSAV_{limit}_{number:04d}.zmx")


def measure_tolerance_lenses(system, zosapi, lens_files, rows, pop_settings_file, pixel_settings,
                             mf_file, label=''):
    """
    Stage B for a list of saved lenses: measures the ZPLM 21 rows on each, then re-saves it with
    the full merit function and a copy of the POP settings file beside it, so the macro works
    when the lens is opened in OpticStudio. Returns {lens path: {row number: value}}.
    """
    results = {}
    for n, path in enumerate(lens_files, start=1):
        if not system.LoadFile(path, False):
            print(f"[Warning] {label}could not open {os.path.basename(path)}; its crosstalk is left blank.")
            continue
        results[path] = measure_crosstalk_rows(system, zosapi, rows, pop_settings_file, pixel_settings)
        system.MFE.LoadMeritFunction(mf_file)
        system.Save()
        shutil.copyfile(pop_settings_file, os.path.splitext(path)[0] + POP_TEXT_SETTINGS_SUFFIX)
        print(f"\r[*] {label}crosstalk measured on {n}/{len(lens_files)} lenses", end='', flush=True)
    print()
    return results


def _stage_b_worker(worker, lens_files, rows, pop_settings_file, pixel_settings, mf_file):
    """Stage B in a separate process with its own OpticStudio instance."""
    zos = PythonStandaloneApplication()
    try:
        return measure_tolerance_lenses(zos.TheSystem, zos.ZOSAPI, lens_files, rows, pop_settings_file,
                                        pixel_settings, mf_file, label=f"worker {worker}: ")
    finally:
        zos.TheApplication.CloseApplication()
        zos.TheApplication = None


def run_stage_b(system, zosapi, lens_files, rows, pop_settings_file, pixel_settings, mf_file, workers):
    """
    Measures crosstalk on every saved lens, in this process (workers = 1) or split across that
    many OpticStudio instances, each in its own process. Returns {lens path: {row: value}}.
    """
    if workers <= 1:
        return measure_tolerance_lenses(system, zosapi, lens_files, rows, pop_settings_file,
                                        pixel_settings, mf_file)
    chunks = [lens_files[w::workers] for w in range(workers)]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_stage_b_worker, w + 1, chunk, rows, pop_settings_file, pixel_settings, mf_file)
                   for w, chunk in enumerate(chunks) if chunk]
        for future in futures:
            results.update(future.result())
    return results


# Time reporting disabled.
# def _format_duration(seconds):
#     """
#     Formats a duration in seconds as m:ss, or h:mm:ss for an hour or more.
#     """
#     minutes, secs = divmod(int(round(seconds)), 60)
#     hours, minutes = divmod(minutes, 60)
#     return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
#
#
# def estimate_tolerance_time(system, eval_seconds, inject_commands):
#     """
#     Prints a rough upfront estimate of the sensitivity run time, rounded up to the next
#     ESTIMATE_STEP_MIN minutes, as a sanity check for the engineer. Without compensators,
#     each tolerance is evaluated at its minimum and maximum, plus one nominal evaluation.
#     """
#     tde_types = [str(system.TDE.GetOperandAt(k).Type) for k in range(1, system.TDE.NumberOfOperands + 1)]
#     num_tolerances = sum(1 for t in tde_types if t not in NON_TOLERANCE_TYPES)
#
#     if any(t in COMPENSATOR_TYPES for t in tde_types):
#         print(f"[*] {num_tolerances} tolerances with compensators: run time can't be estimated upfront.")
#         return
#
#     estimate_min = (1 + 2 * num_tolerances) * eval_seconds / 60
#     step = ESTIMATE_STEP_MIN
#     print(f"[*] {num_tolerances} tolerances: expected to take less than {math.ceil(estimate_min / step) * step} minutes.")
#     if inject_commands:
#         print("[*] Note: additional script commands (e.g. compensation) can make the run take longer.")


def create_tol_script(zemax_data_dir, mf_file, operand_labels, inject_commands, xtra_script_path):
    """
    Writes a Zemax Tolerance Script (*.TSC) that reports each merit function operand
    as its own criterion. The script goes in the Tolerance folder of OpticStudio's
    current data folder, which is where the Tolerancing tool looks for it.
    Returns the script path and the CSV header names.
    """
    zemax_tol_folder = os.path.join(zemax_data_dir, "Tolerance")
    quick_tol_script = os.path.join(zemax_tol_folder, 'OptiSenseScript.tsc')
    header_names = LABEL_COLUMNS + operand_labels

    with open(quick_tol_script, "w") as file:
        if inject_commands:
            with open(xtra_script_path, 'r', encoding='utf-8') as sfile:
                extra_commands = sfile.read()
            # Ensure the next command starts on its own line even if the file lacks a final newline
            file.write(extra_commands if extra_commands.endswith('\n') else extra_commands + '\n')

        file.write("! Evaluate system with given merit function.\n")
        file.write(f"LOADMERIT \"{mf_file}\"\n")
        file.write("GETMERIT\n")
        file.write("FORMAT 1.6 EXP\n\n")

        # Write the formatted REPORT_NB lines
        for i, label in enumerate(operand_labels, start=1):
            file.write(f"REPORT_NB \"{label}\" {i}\n")

    print(f"[*] Tolerance Script saved to {zemax_tol_folder}")
    return quick_tol_script, header_names


def _set_tolerancing_cores(tol, cores):
    """
    Limits the Tolerancing tool to the given number of CPU cores (None keeps OpticStudio's
    setting). The choices are read from OpticStudio, matching on the number in each label.
    """
    if cores is None:
        print(f"[*] Tolerancing cores: OpticStudio setting ({tol.SetupCore})")
        return
    choices = [str(tol.GetSetupCoreAt(i)) for i in range(tol.NumberOfSetupCores)]
    numbers = [int(m.group()) if (m := re.search(r'\d+', c)) else None for c in choices]
    if cores in numbers:
        tol.SetupCoreIndex = numbers.index(cores)
        print(f"[*] Tolerancing cores: {cores}")
    else:
        print(f"[Warning] {cores} cores is not one of OpticStudio's choices ({', '.join(choices)}); "
              f"using its setting ({tol.SetupCore}).")


def run_tolerance_analysis(system, zosapi, tol_script, ztd_filename, txt_filename, cores=None, progress=None):
    """
    Configures and executes the tolerance analysis using the generated script.
    progress: optional function returning a one-line status, printed about once a second.
    """
    tol = system.Tools.OpenTolerancing()
    if tol is None:
        raise RuntimeError("Could not open the Tolerancing tool. Is another analysis tool already open?")

    try:
        tol.SetupMode = zosapi.Tools.Tolerancing.SetupModes.Sensitivity
        tol.Criterion = zosapi.Tools.Tolerancing.Criterions.UserScript
        tol.SaveTolDataFile = True
        tol.TolDataFile = ztd_filename
        tol.OutputFile = txt_filename
        tol.NumberOfRuns = 0
        tol.NumberToSave = 0
        _set_tolerancing_cores(tol, cores)

        # Search for tolerance script and assign it
        tol_idx = None
        script_basename = os.path.basename(tol_script).lower()
        for i in range(tol.NumberOfCriterionScripts):
            if script_basename == tol.GetCriterionScriptAt(i).lower():
                tol_idx = i
                break

        if tol_idx is None:
            raise ValueError(f"Could not find tolerance script '{script_basename}' in Zemax.")

        tol.CriterionScript = tol_idx

        # Time reporting disabled; run with a single blocking call instead.
        # # Run in the background and show elapsed time; the sleep keeps polling overhead negligible
        # print("[*] Tolerance analysis running. Press Ctrl+C to cancel.")
        # start = time.perf_counter()
        # tol.Run()
        # try:
        #     while tol.IsRunning:
        #         time.sleep(PROGRESS_INTERVAL_S)
        #         print(f"\r[*] Elapsed: {_format_duration(time.perf_counter() - start)}", end="", flush=True)
        # except KeyboardInterrupt:
        #     tol.Cancel()
        #     raise
        # tol.WaitForCompletion()
        # print(f"\n[*] Tolerance analysis finished in {_format_duration(time.perf_counter() - start)}")
        # Run in the background so Ctrl+C can cancel: WaitWithTimeout returns about once a
        # second, which gives Python a chance to act on the keypress
        print("[*] Tolerance analysis in progress. This may take a while... (press Ctrl+C to cancel)")
        tol.Run()
        try:
            while tol.IsRunning:
                tol.WaitWithTimeout(CANCEL_CHECK_S)
                if progress is not None:
                    print(f"\r{progress()}", end="", flush=True)
            if progress is not None:
                print()
        except KeyboardInterrupt:
            print("\n[*] Cancelling the tolerance analysis...")
            tol.Cancel()
            raise

        if not tol.Succeeded:
            raise RuntimeError(f"Tolerance analysis did not complete: {tol.ErrorMessage}")
    finally:
        tol.Close()


def _evaluation_failed(value, is_macro):
    """
    True if a criterion value signals a failed evaluation: non-finite for any operand, or
    exactly 0 (macro error) or -1 (macro's chief-ray-failure code) for a ZPLM operand.
    """
    return not math.isfinite(value) or (is_macro and value in MACRO_FAILURE_VALUES)


def _larger_change(value_min, value_max, nominal, is_macro):
    """
    Returns the change from nominal at the tolerance's minimum or maximum, whichever is
    larger in magnitude, keeping its sign. Returns NaN if either evaluation failed.
    """
    if _evaluation_failed(value_min, is_macro) or _evaluation_failed(value_max, is_macro):
        return math.nan
    change_min = value_min - nominal
    change_max = value_max - nominal
    return change_min if abs(change_min) > abs(change_max) else change_max


def _format_cell(value, failed_text=''):
    """
    Formats a CSV cell: floats in scientific notation, NaN as failed_text, anything else as-is.
    """
    if isinstance(value, float):
        return failed_text if math.isnan(value) else f"{value:.6e}"
    return value


def read_sensitivity_data(system, ztd_filepath, num_operands):
    """
    Opens the ZTD file and returns one dict per tolerance, in Tolerance Data Editor order:
    type, comment, minimum, maximum, and at_min / at_max, the value of every merit function
    operand at each tolerance limit (EstimatedChangeMinimum/Maximum hold those values).
    """
    tol_data_viewer = system.Tools.OpenToleranceDataViewer()
    if tol_data_viewer is None:
        raise RuntimeError("Could not open the Tolerance Data Viewer. Is another analysis tool already open?")

    try:
        tol_data_viewer.FileName = ztd_filepath
        tol_data_viewer.RunAndWaitForCompletion()
        if not tol_data_viewer.Succeeded:
            raise RuntimeError(f"Could not read tolerance data file {ztd_filepath}: {tol_data_viewer.ErrorMessage}")
        sdata = tol_data_viewer.SensitivityData

        # Criterion 0 is skipped; criteria 1..N correspond to the REPORT_NB lines (MFE operands 1..N)
        if sdata.NumberOfCriteria - 1 != num_operands:
            raise ValueError(
                f"Tolerance data has {sdata.NumberOfCriteria - 1} report criteria, "
                f"but the merit function has {num_operands} operands."
            )

        tolerances = []
        for m in range(sdata.NumberOfResultOperands):
            sd_op = sdata.GetOperand(m)
            effects = [sd_op.GetEffectOnCriterion(j) for j in range(1, num_operands + 1)]
            tolerances.append({
                'type': str(sd_op.OperandType), 'comment': str(sd_op.Comment),
                'minimum': sd_op.Minimum, 'maximum': sd_op.Maximum,
                'at_min': [e.EstimatedChangeMinimum for e in effects],
                'at_max': [e.EstimatedChangeMaximum for e in effects],
            })
    finally:
        tol_data_viewer.Close()
    return tolerances


def merge_stage_b(tolerances, expected, rows, stage_b, lens_folder):
    """
    Puts stage B's ZPLM 21 values into each tolerance's at_min / at_max, from the lenses its
    SAVE row wrote. expected is [(type, comment)] from add_save_rows, checked against the
    tolerance data so a lens can never be matched to the wrong tolerance. A missing lens
    leaves NaN (reported as a failed evaluation).
    """
    found = [(t['type'], t['comment']) for t in tolerances]
    if found != expected:
        raise RuntimeError("The tolerance data does not list the tolerances in the order OptiSense saved "
                           "their lenses, so stage B results can't be matched to them.")
    by_path = {os.path.normcase(path): values for path, values in stage_b.items()}
    for number, tol in enumerate(tolerances, start=1):
        for limit, key in (('MIN', 'at_min'), ('MAX', 'at_max')):
            values = by_path.get(os.path.normcase(tolerance_lens_path(lens_folder, number, limit)))
            for row in rows:
                tol[key][row.index - 1] = values[row.index] if values else math.nan


def export_results(tolerances, nom_mf_results, header_names, output_csv_path):
    """
    Computes each tolerance's change in every operand, the Zeiss summaries, and exports to CSV.
    """
    num_operands = len(nom_mf_results)
    operand_labels = header_names[len(LABEL_COLUMNS):]
    is_macro = [label.startswith('ZPLM') for label in operand_labels]

    tol_rows = []
    for tol in tolerances:
        effects = [_larger_change(tol['at_min'][j], tol['at_max'][j], nom_mf_results[j], is_macro[j])
                   for j in range(num_operands)]
        tol_rows.append([tol['type'], tol['comment'], tol['minimum'], tol['maximum']] + effects)

    # Summary Calculations: Abs sum, RSS, Raentsch sum, skipping failed evaluations (NaN)
    effects_array = np.array([row[len(LABEL_COLUMNS):] for row in tol_rows], dtype=float).reshape(-1, num_operands)
    failed = np.isnan(effects_array)
    col_abs_sum = np.nansum(np.abs(effects_array), axis=0)
    col_rss = np.sqrt(np.nansum(np.square(effects_array), axis=0))
    col_raentsch = np.sqrt(col_abs_sum * col_rss)
    col_failed = failed.sum(axis=0)

    for j in np.flatnonzero(col_failed):
        rows = ", ".join(f"tolerance #{m + 1} ({tol_rows[m][0]})" for m in np.flatnonzero(failed[:, j]))
        print(f"[Warning] Failed evaluations excluded from '{operand_labels[j]}' (column {len(LABEL_COLUMNS) + j + 1}): {rows}")
    for j, label in enumerate(operand_labels):
        if _evaluation_failed(nom_mf_results[j], is_macro[j]):
            print(f"[Warning] Nominal value of '{label}' (column {len(LABEL_COLUMNS) + j + 1}) looks like a failed evaluation; check this column.")

    summary_rows = [
        ['', 'Nominal Values', '', ''] + list(nom_mf_results),
        ['', 'Sum of Absolute Values', '', ''] + col_abs_sum.tolist(),
        ['', 'Root Sum Square', '', ''] + col_rss.tolist(),
        ['', 'Raentsch Sum (Zeiss Approach)', '', ''] + col_raentsch.tolist(),
        ['', 'Failed Evaluations (excluded above)', '', ''] + [int(n) for n in col_failed],
        [''] * len(header_names),  # blank separator row
    ]
    n_labels = len(LABEL_COLUMNS)
    formatted = ([row[:n_labels] + [_format_cell(v) for v in row[n_labels:]] for row in summary_rows]
                 + [row[:n_labels] + [_format_cell(v, FAILED_TEXT) for v in row[n_labels:]] for row in tol_rows])

    final_df = pd.DataFrame(formatted, columns=header_names)
    final_df.to_csv(output_csv_path, sep=',', mode='w', index=False, header=True)

    print(f"[*] Analysis complete! Results saved to: {output_csv_path}")


def main():
    # start_time = time.perf_counter()

    # 1. Load Configuration
    config = load_configuration()
    zemax_file = os.path.normpath(config.get('PATHS', 'ZemaxFile'))
    mf_file = os.path.normpath(config.get('PATHS', 'MeritFunctionFile'))
    output_dir = os.path.normpath(config.get('PATHS', 'OutputDirectory'))

    inject_commands = config.getboolean('SETTINGS', 'InjectAdditionalCommands', fallback=False)
    xtra_script_file = os.path.normpath(config.get('SETTINGS', 'AdditionalCommandsFile'))

    # Cores: a number limits tolerancing to that many CPU cores; blank keeps OpticStudio's setting
    cores_text = config.get('SETTINGS', 'Cores', fallback=str(DEFAULT_CORES)).strip()
    if cores_text and not cores_text.isdigit():
        print(f"[Error] Cores in config.ini must be a whole number or blank, not '{cores_text}'")
        return 1
    cores = int(cores_text) if cores_text else None
    # Only safe with a macro that never shares a file between evaluations (ZPL21 as of 2026-10-30)
    parallel_macros = config.getboolean('SETTINGS', 'ParallelMacros', fallback=False)
    # python: two-stage run with ZPLM 21 measured in Python; zpl: the macro runs inside tolerancing
    crosstalk_mode = config.get('SETTINGS', 'CrosstalkMode', fallback='python').strip().lower()
    if crosstalk_mode not in CROSSTALK_MODES:
        print(f"[Error] CrosstalkMode in config.ini must be one of {', '.join(CROSSTALK_MODES)}, not '{crosstalk_mode}'")
        return 1
    # OpticStudio instances measuring crosstalk in stage B (each needs a licence seat)
    workers_text = config.get('SETTINGS', 'CrosstalkWorkers', fallback='1').strip()
    if not workers_text.isdigit() or int(workers_text) < 1:
        print(f"[Error] CrosstalkWorkers in config.ini must be a whole number of at least 1, not '{workers_text}'")
        return 1
    workers = int(workers_text)

    # Validate input files before launching Zemax
    required_files = [('Zemax file', zemax_file), ('Merit function file', mf_file)]
    if inject_commands:
        required_files.append(('Additional commands file', xtra_script_file))
    missing = [f"{name} not found: {path}" for name, path in required_files if not os.path.exists(path)]
    if missing:
        for message in missing:
            print(f"[Error] {message}")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = os.path.splitext(os.path.basename(zemax_file))[0]

    ztd_filename = f"{filename_prefix}_ZTDFile.ZTD"
    txt_filename = f"{filename_prefix}_TolSummary.TXT"
    ztd_filepath = os.path.join(os.path.dirname(zemax_file), ztd_filename)
    txt_filepath = os.path.join(os.path.dirname(zemax_file), txt_filename)
    output_csv_path = os.path.join(output_dir, f"{filename_prefix}_TolResults.csv")
    lens_dir = os.path.dirname(zemax_file)
    tol_lens_dir = os.path.join(output_dir, filename_prefix + TOL_LENS_FOLDER_SUFFIX)
    stage_a_mf = os.path.join(tempfile.gettempdir(), f"OptiSense_StageA_{os.getpid()}.MF")

    exit_code = 0
    zos = None
    owns_lens_dir_tsavs = False
    try:
        # Remove results from any previous run so stale data can never be read back in
        for old_file in (ztd_filepath, txt_filepath, output_csv_path):
            if os.path.exists(old_file):
                os.remove(old_file)

        # 2. Initialize Zemax
        print("[*] Connecting to Zemax OpticStudio...")
        zos = PythonStandaloneApplication()
        zosapi = zos.ZOSAPI
        system = zos.TheSystem
        if not system.LoadFile(zemax_file, False):
            raise RuntimeError(f"OpticStudio could not load the lens file: {zemax_file}")

        # The pixel-readout macro needs its POP settings file before the merit function is evaluated
        calls_macros = merit_function_calls_macros(mf_file)
        pop_settings_file = None
        if calls_macros:
            pop_settings_file = write_pop_text_settings(system, zosapi, zemax_file)
            print(f"[*] POP settings for the macro saved to {pop_settings_file}")

        # 3. Load Merit Function and record nominal values
        print("[*] Calculating Nominal Merit Function...")
        # nom_mf_results, operand_labels, eval_seconds = load_merit_function(system, zosapi, mf_file)
        # estimate_tolerance_time(system, eval_seconds, inject_commands)
        nom_mf_results, operand_labels = load_merit_function(system, zosapi, mf_file)

        # Two-stage run when the merit function measures crosstalk with ZPLM 21 and no other macro
        crosstalk_rows, other_macros = read_macro_rows(system, zosapi) if calls_macros else ([], [])
        two_stage = bool(crosstalk_rows) and crosstalk_mode == 'python'
        if two_stage and other_macros:
            print(f"[*] The merit function also calls macro(s) {', '.join(map(str, other_macros))}, which only "
                  "run inside tolerancing; running in a single stage with every macro (CrosstalkMode = zpl).")
            two_stage = False

        tol_merit_function = mf_file
        if two_stage:
            print(f"[*] Two-stage run: {len(crosstalk_rows)} ZPLM {CROSSTALK_MACRO} rows measured in Python.")
            macro_path = os.path.join(str(zos.TheApplication.ZemaxDataDir), 'Macros', f'ZPL{CROSSTALK_MACRO}.ZPL')
            pixel_settings = PixelSettings.from_zpl(macro_path)

            # Lenses from an earlier run in the lens folder would be mistaken for this run's
            stale = glob.glob(os.path.join(lens_dir, TOL_LENS_GLOB))
            if stale:
                raise RuntimeError(f"{len(stale)} TSAV_*.zmx files from an earlier run are in {lens_dir}. "
                                   "Move or delete them first; OptiSense saves this run's lenses there "
                                   "before moving them to the results folder.")
            # From here on every TSAV file in the lens folder is this run's; removed on exit if left behind
            owns_lens_dir_tsavs = True
            os.makedirs(tol_lens_dir, exist_ok=True)
            for old_file in glob.glob(os.path.join(tol_lens_dir, 'TSAV_*')):
                os.remove(old_file)

            # Nominal crosstalk from Python, checked against the macro's
            python_nominal = measure_crosstalk_rows(system, zosapi, crosstalk_rows, pop_settings_file, pixel_settings)
            check_crosstalk_against_macro(nom_mf_results, python_nominal, operand_labels)
            for index, value in python_nominal.items():
                nom_mf_results[index - 1] = value

            write_stage_a_merit_function(system, zosapi, crosstalk_rows, stage_a_mf)
            tol_merit_function = stage_a_mf
            saved_tolerances = add_save_rows(system, zosapi)

        # 4. Create Tolerance Script
        print("[*] Generating Tolerance Script...")
        quick_tol_script, header_names = create_tol_script(
            str(zos.TheApplication.ZemaxDataDir), tol_merit_function, operand_labels, inject_commands, xtra_script_file
        )

        # 5. Run Tolerance Analysis
        # A macro that writes a shared file (e.g. a POP beam file) collides with itself when
        # evaluations run in parallel, so macros run on 1 core unless ParallelMacros says otherwise
        if calls_macros and not two_stage and cores != 1 and not parallel_macros:
            print("[*] Merit function calls a ZPLM macro; running tolerancing on 1 core. Set "
                  "ParallelMacros = True in config.ini if every macro is safe to run in parallel.")
            cores = 1
        progress = None
        if two_stage:
            print("[*] Stage A: tolerancing without the crosstalk macro, saving every perturbed lens.")
            expected = 2 * len(saved_tolerances)
            progress = lambda: (f"[*] Stage A: {len(glob.glob(os.path.join(lens_dir, TOL_LENS_GLOB)))} "
                                f"of {expected} lenses saved")
        run_tolerance_analysis(system, zosapi, quick_tol_script, ztd_filename, txt_filename, cores, progress)

        # 6. Process and Export Results
        print("[*] Processing Tolerance Data Viewer results...")
        tolerances = read_sensitivity_data(system, ztd_filepath, len(nom_mf_results))

        if two_stage:
            for path in glob.glob(os.path.join(lens_dir, TOL_LENS_GLOB)):
                shutil.move(path, os.path.join(tol_lens_dir, os.path.basename(path)))
            lens_files = sorted(glob.glob(os.path.join(tol_lens_dir, TOL_LENS_GLOB)))
            if len(lens_files) != expected:
                print(f"[Warning] Stage A saved {len(lens_files)} of {expected} lenses; tolerances without "
                      "a lens are reported as failed in the crosstalk columns.")
            print(f"[*] Stage B: measuring crosstalk on {len(lens_files)} lenses "
                  f"({workers} OpticStudio instance{'s' if workers > 1 else ''}).")
            stage_b = run_stage_b(system, zosapi, lens_files, crosstalk_rows, pop_settings_file,
                                  pixel_settings, mf_file, workers)
            merge_stage_b(tolerances, saved_tolerances, crosstalk_rows, stage_b, tol_lens_dir)
            print(f"[*] Tolerance lenses (reopen in OpticStudio to inspect) saved to: {tol_lens_dir}")

        export_results(tolerances, nom_mf_results, header_names, output_csv_path)

    except KeyboardInterrupt:
        print("[*] Run cancelled. No results were written.")
        exit_code = 1

    except Exception:
        print("\n[Fatal Error] The analysis did not complete. No results were written.")
        traceback.print_exc()
        exit_code = 1

    finally:
        # 7. Safe Cleanup
        if os.path.exists(stage_a_mf):
            os.remove(stage_a_mf)
        # Lenses a cancelled or failed stage A left in the lens folder would block the next run
        if owns_lens_dir_tsavs:
            for path in glob.glob(os.path.join(lens_dir, TOL_LENS_GLOB)):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"[Warning] Could not remove {path}: {e}")
        if zos is not None and zos.TheApplication is not None:
            try:
                zos.TheApplication.CloseApplication()
                zos.TheApplication = None
                print("[*] Zemax connection successfully closed.")
            except Exception as e:
                print(f"[Warning] Could not close OpticStudio cleanly; check Task Manager for a leftover instance: {e}")

        # elapsed_time = time.perf_counter() - start_time
        # print(f"[*] Total script execution time: {elapsed_time / 60:.2f} minutes")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
