import numpy as np

npz_path = "FuzzySugeno_e_Theta1.npz"
out_prefix = "FuzzySugeno_e_Theta1"

data = np.load(npz_path)

# Save vectors
np.savetxt(f"{out_prefix}_evec.csv",  data["evec_Theta1"],  delimiter=",")
np.savetxt(f"{out_prefix}_devec.csv", data["devec_Theta1"], delimiter=",")

# Save matrices
np.savetxt(f"{out_prefix}_Ymat_dKp.csv", data["Ymat_dKp_Theta1"], delimiter=",")
np.savetxt(f"{out_prefix}_Ymat_dKd.csv", data["Ymat_dKd_Theta1"], delimiter=",")

# If you prefer TXT (space-separated), set delimiter=" "