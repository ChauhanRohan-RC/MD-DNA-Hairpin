import matplotlib.pyplot as plt
import pandas as pd

import C
from C import find_overlap_y_diff

if __name__ == '__main__':
    # INPUT -----------------------------
    set1_file_name = "sp_app-fit-1.2.sim_app_pmf_aligned.csv"
    set2_file_name = "sp_app-fit-2.2.sim_app_pmf_aligned.csv"
    col_name_x = C.COL_NAME_EXTENSION
    col_name_y = C.COL_NAME_PMF_RECONSTRUCTED

    # OUTPUT ----------------------------
    set2_aligned_out_file_name = "sp_app-fit-2.2.sim_app_pmf_aligned.2.csv"

    # ------------------------------------

    set1_df = pd.read_csv(set1_file_name, sep=r"\s+", comment="#")
    set2_df = pd.read_csv(set2_file_name, sep=r"\s+", comment="#")
    x1 = set1_df[col_name_x].values
    y1 = set1_df[col_name_y].values

    x2 = set2_df[col_name_x].values
    y2 = set2_df[col_name_y].values

    diff = find_overlap_y_diff(x1, y1, x2, y2)
    if diff is not None:
        y2 -= diff

    set2_df[col_name_y] = y2
    set2_df.to_csv(set2_aligned_out_file_name, sep="\t", header=True, index=False, index_label=False)

    # PLOT data
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.show()
