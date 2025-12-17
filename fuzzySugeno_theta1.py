import PoE.controller as controller
import PoE.helper as helper

theta_N = 1
npz_path = f"FuzzyLogicOut/FuzzySugeno_e_Theta_{theta_N}.npz"

e_range  = (-0.1, 0.1)
de_range = (-1.0, 1.0)
 
ruleTable_dKp = [
    ["L","L","M","M","S","ZO","ZO"],
    ["L","M","M","S","ZO","S","ZO"],
    ["M","M","S","ZO","S","M","M"],
    ["M","S","ZO","ZO","ZO","S","M"],
    ["M","S","ZO","S","M","M","L"],
    ["ZO","S","M","S","M","L","L"],
    ["ZO","ZO","M","M","L","L","L"],
]
ruleTable_dKd = [
    ["L","M","M","S","M","M","L"],
    ["M","S","S","ZO","S","S","M"],
    ["M","S","ZO","ZO","ZO","S","M"],
    ["S","ZO","ZO","ZO","ZO","ZO","S"],
    ["M","S","ZO","ZO","ZO","S","M"],
    ["M","S","S","ZO","S","S","M"],
    ["L","M","M","S","M","M","L"],
]

fs = controller.FuzzySugeno(e_range, de_range, ruleTable_dKp, ruleTable_dKd)

fs.save_npz(theta_N=theta_N, fileNamePath=npz_path or ".", nE=101, nDE=101)

helper.plot_3d_heatmaps(theta_N=theta_N, npz_path=npz_path)