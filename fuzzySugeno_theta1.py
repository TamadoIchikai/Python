import PoE.controller as controller
import PoE.helper as helper

theta_N = 1
npz_path = f"FuzzyLogicOut/FuzzySugeno_e_Theta_{theta_N}.npz"

e_range  = (-0.1, 0.1)
de_range = (-1.0, 1.0)

# MAP_VAL from mainFPID.py with 7-label support
MAP_VAL = {"NL": -1.00, "NM": -0.66, "NS": -0.33, "ZO": 0.00, "PS": 0.33, "PM": 0.66, "PL": 1.00}
 
ruleTable_dKp = [
    ["PL", "PL", "PM", "PM", "PS", "ZO", "ZO"],
    ["PL", "PM", "PM", "PS", "ZO", "NS", "ZO"],
    ["PM", "PM", "PS", "ZO", "NS", "NS", "NM"],
    ["PM", "PS", "ZO", "NL", "ZO", "PS", "PM"],
    ["NM", "NS", "NS", "ZO", "PS", "PM", "PM"],
    ["ZO", "NS", "ZO", "PS", "PM", "PM", "PL"],
    ["ZO", "ZO", "PS", "PM", "PM", "PL", "PL"],
]
ruleTable_dKd = [
    ["PM", "PM", "PM", "PS", "PM", "PM", "PM"],
    ["PM", "PS", "PS", "ZO", "PS", "PS", "PM"],
    ["PS", "ZO", "NS", "NL", "NS", "ZO", "PS"],
    ["ZO", "NL", "NL", "NL", "NL", "NL", "ZO"],
    ["PS", "ZO", "NS", "NL", "NS", "ZO", "PS"],
    ["PM", "PS", "PS", "ZO", "PS", "PS", "PM"],
    ["PM", "PM", "PM", "PS", "PM", "PM", "PM"],
]

fs = controller.FuzzySugeno(e_range, de_range, ruleTable_dKp, ruleTable_dKd, map_val=MAP_VAL)

fs.save_npz(theta_N=theta_N, fileNamePath=npz_path or ".", nE=501, nDE=501)

helper.plot_3d_heatmaps(theta_N=theta_N, npz_path=npz_path)
