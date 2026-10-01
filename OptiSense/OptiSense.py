# OptiSense: automated tolerance sensitivity analysis for Zemax OpticStudio via the ZOS-API.
#
# Given a lens file (.zmx) and a merit function (.mf), this script:
#   1. Loads the lens and merit function in OpticStudio and records the nominal operand values.
#   2. Writes a tolerance script (QuickTolScript.TSC) that reports every merit function operand
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
import pandas as pd
import numpy as np

# Ensure PythonStandaloneApplication is in the same directory or Python PATH
from PythonStandaloneApplication import PythonStandaloneApplication

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_COLUMNS = ['Operand', 'Comment', 'Min Tol', 'Max Tol']


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
    if not os.path.exists(mf_file):
        raise FileNotFoundError(f"Merit function file not found: {mf_file}")

    system.MFE.LoadMeritFunction(mf_file)
    system.MFE.CalculateMeritFunction()

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

    return nom_mf_results, operand_labels


def create_tol_script(mf_file, operand_labels, inject_commands, xtra_script_path):
    """
    Writes a Zemax Tolerance Script (*.TSC) that reports each merit function operand
    as its own criterion. Returns the script path and the CSV header names.
    """
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    zemax_tol_folder = os.path.join(documents_path, "Zemax", "Tolerance")
    quick_tol_script = os.path.join(zemax_tol_folder, 'QuickTolScript.TSC')
    header_names = LABEL_COLUMNS + operand_labels

    with open(quick_tol_script, "w") as file:
        if inject_commands and os.path.exists(xtra_script_path):
            with open(xtra_script_path, 'r', encoding='utf-8') as sfile:
                file.writelines(sfile.readlines())

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

    print("[*] Tolerance analysis in progress. This may take a while...")
    tol.RunAndWaitForCompletion()
    tol.Close()


def process_and_export_results(system, ztd_filepath, nom_mf_results, header_names, output_csv_path):
    """
    Opens the ZTD file, extracts sensitivity data, computes Zeiss math, and exports to CSV.
    """
    tol_data_viewer = system.Tools.OpenToleranceDataViewer()
    tol_data_viewer.FileName = ztd_filepath
    tol_data_viewer.RunAndWaitForCompletion()
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

    # Validate core file paths before launching Zemax
    if not os.path.exists(zemax_file):
        print(f"[Error] Zemax file not found: {zemax_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = os.path.splitext(os.path.basename(zemax_file))[0]

    zos = None
    try:
        # 2. Initialize Zemax
        print("[*] Connecting to Zemax OpticStudio...")
        zos = PythonStandaloneApplication()
        zosapi = zos.ZOSAPI
        system = zos.TheSystem
        system.LoadFile(zemax_file, False)

        # 3. Load Merit Function and record nominal values
        print("[*] Calculating Nominal Merit Function...")
        nom_mf_results, operand_labels = load_merit_function(system, zosapi, mf_file)

        # 4. Create Tolerance Script
        print("[*] Generating Tolerance Script...")
        quick_tol_script, header_names = create_tol_script(
            mf_file, operand_labels, inject_commands, xtra_script_file
        )

        # 5. Run Tolerance Analysis
        ztd_filename = f"{filename_prefix}_ZTDFile.ZTD"
        txt_filename = f"{filename_prefix}_TolSummary.TXT"
        run_tolerance_analysis(system, zosapi, quick_tol_script, ztd_filename, txt_filename)

        # 6. Process and Export Results
        print("[*] Processing Tolerance Data Viewer results...")
        ztd_filepath = os.path.join(os.path.dirname(zemax_file), ztd_filename)
        output_csv_path = os.path.join(output_dir, f"{filename_prefix}_TolResults.csv")

        process_and_export_results(
            system, ztd_filepath, nom_mf_results, header_names, output_csv_path
        )

    except Exception as e:
        print(f"\n[Fatal Error] An unexpected error occurred: {e}")

    finally:
        # 7. Safe Cleanup
        if zos is not None:
            del zos
            print("[*] Zemax connection successfully closed.")

        elapsed_time = time.perf_counter() - start_time
        print(f"[*] Total script execution time: {elapsed_time / 60:.2f} minutes")


if __name__ == '__main__':
    main()
