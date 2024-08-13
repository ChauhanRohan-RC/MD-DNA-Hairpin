import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sp_impl


def find_overlap_y_diff(x1, y1, x2, y2):
    """
    Returns the avg difference (y2 - y1) between the overlapping regions of x1 and x2
    Assumes x1 and x2 are sorted in ascending order and are overlapped
    """
    # ALign overlapping regions of pmf1 and pmf2
    x1_max = x1[-1]
    x2_min = x2[0]

    x1_start_idx = np.searchsorted(x1, x2_min) + 1
    x2_end_idx = np.searchsorted(x2, x1_max) - 1
    common_len = min(x2_end_idx, len(y1) - x1_start_idx)

    return np.sum(y2[:common_len] - y1[x1_start_idx: x1_start_idx + common_len]) / common_len


# INPUT -----------------------------
set1_file_name = "sp_app-fit-1.2.sim_app_pmf_aligned.csv"
set2_file_name = "sp_app-fit-2.2.sim_app_pmf_aligned.csv"
col_name_x = sp_impl.COL_NAME_EXTENSION
col_name_y = sp_impl.COL_NAME_PMF_RE

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
y2 -= diff

set2_df[col_name_y] = y2
set2_df.to_csv(set2_aligned_out_file_name, sep="\t", header=True, index=False, index_label=False)

# PLOT data
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.show()
