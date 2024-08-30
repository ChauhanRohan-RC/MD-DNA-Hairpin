import os.path

from C import *
from double_well_pmf import analyze_pmf_min_max

pmf_df_file_name = "results-theory_sim/sp_app/sp_app-fit-2.2.sim_app_pmf_aligned.csv"
x_col_name = COL_NAME_EXTENSION
pmf_col_name = COL_NAME_PMF_RECONSTRUCTED

out_file_name = "results-theory_sim/sp_app/sim_app_pmf_aligned2.2.pmf_re_min_max.txt"

analyze_pmf_min_max(pmf_df_file_name,
                    x_col_name=x_col_name,
                    pmf_col_name=pmf_col_name,
                    out_file_name=out_file_name)