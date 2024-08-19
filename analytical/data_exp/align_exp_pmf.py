import sys

import pandas as pd
import numpy as np
from C import read_csv, to_csv, COL_NAME_X, COL_NAME_PMF

input_exp_pmf_file_name = "pmf-2.1.csv"
input_pmf_col_name = COL_NAME_PMF
pmf_offset = -4.890036765764929

output_file_name = "pmf-2.1.csv"

# Main------------------
df = read_csv(input_exp_pmf_file_name)
df[input_pmf_col_name] += pmf_offset
to_csv(df, output_file_name)