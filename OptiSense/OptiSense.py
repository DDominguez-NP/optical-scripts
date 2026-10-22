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
# File paths and options are read from config.ini, located next to this script.

import os
import sys
import time
import math
import configparser
import traceback
import pandas as pd
import numpy as np

# Ensure PythonStandaloneApplication is in the same directory or Python PATH
from PythonStandaloneApplication import PythonStandaloneApplication

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_COLUMNS = ['Operand', 'Comment', 'Min Tol', 'Max Tol']
PROGRESS_INTERVAL_S = 3
ESTIMATE_STEP_MIN = 15

# Tolerance Data Editor rows that don't perturb the lens, so they add no merit function evaluations
COMPENSATOR_TYPES = {'COMP', 'CPAR', 'CMCO', 'CEDV'}
NON_TOLERANCE_TYPES = COMPENSATOR_TYPES | {'NULL', 'TOFF', 'COMM', 'SAVE', 'SEED', 'STAT', 'TWAV'}


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
            'AdditionalCommandsFile': r'C:\Path\To\XtraScript.txt'
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
    eval_start = time.perf_counter()
    system.MFE.CalculateMeritFunction()
    eval_seconds = time.perf_counter() - eval_start

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

    return nom_mf_results, operand_labels, eval_seconds


def _format_duration(seconds):
    """
    Formats a duration in seconds as m:ss, or h:mm:ss for an hour or more.
    """
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def estimate_tolerance_time(system, eval_seconds, inject_commands):
    """
    Prints a rough upfront estimate of the sensitivity run time, rounded up to the next
    ESTIMATE_STEP_MIN minutes, as a sanity check for the engineer. Without compensators,
    each tolerance is evaluated at its minimum and maximum, plus one nominal evaluation.
    """
    tde_types = [str(system.TDE.GetOperandAt(k).Type) for k in range(1, system.TDE.NumberOfOperands + 1)]
    num_tolerances = sum(1 for t in tde_types if t not in NON_TOLERANCE_TYPES)

    if any(t in COMPENSATOR_TYPES for t in tde_types):
        print(f"[*] {num_tolerances} tolerances with compensators: run time can't be estimated upfront.")
        return

    estimate_min = (1 + 2 * num_tolerances) * eval_seconds / 60
    step = ESTIMATE_STEP_MIN
    print(f"[*] {num_tolerances} tolerances: expected to take less than {math.ceil(estimate_min / step) * step} minutes.")
    if inject_commands:
        print("[*] Note: additional script commands (e.g. compensation) can make the run take longer.")


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


def run_tolerance_analysis(system, zosapi, tol_script, ztd_filename, txt_filename):
    """
    Configures and executes the tolerance analysis using the generated script.
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

        # Run in the background and show elapsed time; the sleep keeps polling overhead negligible
        print("[*] Tolerance analysis running. Press Ctrl+C to cancel.")
        start = time.perf_counter()
        tol.Run()
        try:
            while tol.IsRunning:
                time.sleep(PROGRESS_INTERVAL_S)
                print(f"\r[*] Elapsed: {_format_duration(time.perf_counter() - start)}", end="", flush=True)
        except KeyboardInterrupt:
            tol.Cancel()
            raise
        tol.WaitForCompletion()
        print(f"\n[*] Tolerance analysis finished in {_format_duration(time.perf_counter() - start)}")

        if not tol.Succeeded:
            raise RuntimeError(f"Tolerance analysis did not complete: {tol.ErrorMessage}")
    finally:
        tol.Close()


def process_and_export_results(system, ztd_filepath, nom_mf_results, header_names, output_csv_path):
    """
    Opens the ZTD file, extracts sensitivity data, computes Zeiss math, and exports to CSV.
    """
    tol_data_viewer = system.Tools.OpenToleranceDataViewer()
    if tol_data_viewer is None:
        raise RuntimeError("Could not open the Tolerance Data Viewer. Is another analysis tool already open?")

    tol_data_viewer.FileName = ztd_filepath
    tol_data_viewer.RunAndWaitForCompletion()
    if not tol_data_viewer.Succeeded:
        error_message = tol_data_viewer.ErrorMessage
        tol_data_viewer.Close()
        raise RuntimeError(f"Could not read tolerance data file {ztd_filepath}: {error_message}")
    sdata = tol_data_viewer.SensitivityData

    # Criterion 0 is skipped; criteria 1..N correspond to the REPORT_NB lines (MFE operands 1..N)
    num_operands = len(nom_mf_results)
    if sdata.NumberOfCriteria - 1 != num_operands:
        tol_data_viewer.Close()
        raise ValueError(
            f"Tolerance data has {sdata.NumberOfCriteria - 1} report criteria, "
            f"but the merit function has {num_operands} operands."
        )

    tol_rows = []
    for m in range(sdata.NumberOfResultOperands):
        sd_op = sdata.GetOperand(m)
        effects = [
            sd_op.GetEffectOnCriterion(j).EstimatedChangeMaximum - nom_mf_results[j - 1]
            for j in range(1, num_operands + 1)
        ]
        tol_rows.append([str(sd_op.OperandType), str(sd_op.Comment), sd_op.Minimum, sd_op.Maximum] + effects)

    tol_data_viewer.Close()

    # Summary Calculations: Abs sum, RSS, Raentsch sum (full precision; formatting happens at export)
    effects_array = np.array([row[len(LABEL_COLUMNS):] for row in tol_rows], dtype=float).reshape(-1, num_operands)
    col_abs_sum = np.abs(effects_array).sum(axis=0)
    col_rss = np.sqrt(np.square(effects_array).sum(axis=0))
    col_raentsch = np.sqrt(col_abs_sum * col_rss)

    summary_rows = [
        ['', 'Nominal Values', '', ''] + list(nom_mf_results),
        ['', 'Sum of Absolute Values', '', ''] + col_abs_sum.tolist(),
        ['', 'Root Sum Square', '', ''] + col_rss.tolist(),
        ['', 'Raentsch Sum (Zeiss Approach)', '', ''] + col_raentsch.tolist(),
        [''] * len(LABEL_COLUMNS) + [np.nan] * num_operands,  # blank separator row
    ]

    final_df = pd.DataFrame(summary_rows + tol_rows, columns=header_names)
    final_df.to_csv(output_csv_path, sep=',', mode='w', index=False, header=True, float_format='%.6e')

    print(f"[*] Analysis complete! Results saved to: {output_csv_path}")


def main():
    start_time = time.perf_counter()

    # 1. Load Configuration
    config = load_configuration()
    zemax_file = os.path.normpath(config.get('PATHS', 'ZemaxFile'))
    mf_file = os.path.normpath(config.get('PATHS', 'MeritFunctionFile'))
    output_dir = os.path.normpath(config.get('PATHS', 'OutputDirectory'))

    inject_commands = config.getboolean('SETTINGS', 'InjectAdditionalCommands', fallback=False)
    xtra_script_file = os.path.normpath(config.get('SETTINGS', 'AdditionalCommandsFile'))

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

    exit_code = 0
    zos = None
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

        # 3. Load Merit Function and record nominal values
        print("[*] Calculating Nominal Merit Function...")
        nom_mf_results, operand_labels, eval_seconds = load_merit_function(system, zosapi, mf_file)
        estimate_tolerance_time(system, eval_seconds, inject_commands)

        # 4. Create Tolerance Script
        print("[*] Generating Tolerance Script...")
        quick_tol_script, header_names = create_tol_script(
            str(zos.TheApplication.ZemaxDataDir), mf_file, operand_labels, inject_commands, xtra_script_file
        )

        # 5. Run Tolerance Analysis
        run_tolerance_analysis(system, zosapi, quick_tol_script, ztd_filename, txt_filename)

        # 6. Process and Export Results
        print("[*] Processing Tolerance Data Viewer results...")
        process_and_export_results(
            system, ztd_filepath, nom_mf_results, header_names, output_csv_path
        )

    except Exception:
        print("\n[Fatal Error] The analysis did not complete. No results were written.")
        traceback.print_exc()
        exit_code = 1

    finally:
        # 7. Safe Cleanup
        if zos is not None and zos.TheApplication is not None:
            try:
                zos.TheApplication.CloseApplication()
                zos.TheApplication = None
                print("[*] Zemax connection successfully closed.")
            except Exception as e:
                print(f"[Warning] Could not close OpticStudio cleanly; check Task Manager for a leftover instance: {e}")

        elapsed_time = time.perf_counter() - start_time
        print(f"[*] Total script execution time: {elapsed_time / 60:.2f} minutes")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
