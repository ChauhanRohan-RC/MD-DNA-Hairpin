import multiprocessing as mp
import time
from typing import Sequence, Literal

import numpy as np
import pandas as pd
import scipy

COMMENT_TOKEN = "#"

# Constants ------------------------
AVOGADRO_NUMBER = 6.02214076e23

CAL_TO_JOULE = 4.184  # 1 cal = 4.184 J
KCAL_PER_MOL_TO_pN_nM = 6.94121554      # 1 kcal/mol = 6.95 pN nm
KCAL_PER_MOL_A2_TO_pN_PER_nM = KCAL_PER_MOL_TO_pN_nM * 100      # 1 kcal/mol/Å**2 = 695 pN/nm

BOLTZMANN_CONST_JOULE_PER_MOL_K = 8.314             # Ideal gas constant (J/mol/K)
BOLTZMANN_CONST_JOULE_PER_MOLECULE_K = 1.380649e-23      # Boltzmann constant (J/molecule/K) = 8.314 / (6.022 x 10^23)
BOLTZMANN_CONST_KCAL_PER_MOL_K = 1.9872036e-3       # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10^3)
BOLTZMANN_CONST_pN_nM_PER_MOLECULE_K = BOLTZMANN_CONST_JOULE_PER_MOLECULE_K * 1e21   # Boltzmann constant (pN nm/molecule/K)

# Columns --------------------------
COL_NAME_X = "X"
COL_NAME_X0 = "X0"
COL_NAME_EXTENSION = "EXT"
COL_NAME_EXT_BIN = "EXT_BIN"
COL_NAME_EXT_BIN_MEDIAN = "EXT_BIN_MED"
COL_NAME_SP_INTEGRAND = "SP_INTEGRAND"
COL_NAME_SP = "SP"
COL_NAME_PMF = "PMF"
COL_NAME_PMF_IMPOSED = "PMF_IM"  # Imposed PMF
COL_NAME_PMF_RECONSTRUCTED = "PMF_RE"  # Reconstructed PMF
COL_NAME_TIME = "T"
COL_NAME_FIRST_PASS_TIME = "FPT"  # First_passage Time
COL_NAME_CONDITIONAL_PROBABILITY = "CP"  # Conditional Probability
COL_NAME_CONDITIONAL_PROBABILITY_INTEGRAL_OVER_X = "CP_INT_X"
COL_NAME_CONDITIONAL_PROBABILITY_INTEGRAL_OVER_TIME = "CP_INT_T"

PMF_FIT_COL_NAME_PARAM = "PARAM"
PMF_FIT_COL_NAME_PARAM_VALUE = "VALUE"
PMF_FIT_COL_NAME_PARAM_STD_DEV = "STD_DEV"

# Multiprocessing --------------------------
CPU_COUNT = mp.cpu_count()
DEFAULT_PROCESS_COUNT = CPU_COUNT - 1


def mp_execute(worker_func, input_arr: np.ndarray, process_count: int, args: tuple = None) -> np.ndarray:
    sample_count = len(input_arr)

    q, r = divmod(sample_count, process_count)
    chunk_size = q if r == 0 else q + 1

    print("---------------------------------------------")
    print(f"# Computing in Multiprocess Mode"
          f"\n -> Target Function: {worker_func.__name__}"
          f"\n -> Total CPU(s): {CPU_COUNT} | Process Count: {process_count}"
          f"\n -> Total Samples: {sample_count} | Samples per Process: {chunk_size}")

    has_args = isinstance(args, tuple)

    time_start = time.time()

    chunks = []
    for i in range(0, sample_count, chunk_size):
        chunk = input_arr[i: min(i + chunk_size, sample_count)]
        if has_args:
            chunks.append((chunk, *args))
        else:
            chunks.append(chunk)

    with mp.Pool(processes=process_count) as pool:
        if has_args:
            res = pool.starmap(worker_func, chunks)
        else:
            res = pool.map(worker_func, chunks)

    time_end = time.time()

    print(f"Time taken: {time_end - time_start:.2f} s")
    print("---------------------------------------------")
    return np.concatenate(res)


# Pandas Dataframe ---------------------------------------------------------
def to_csv(df: pd.DataFrame, path_or_buf, sep: str = "\t", header=True, index=False, index_label=False, mode: str = "w",
           *args, **kwargs):
    df.to_csv(path_or_buf, sep=sep, header=header, index=index, index_label=index_label, mode=mode, *args, **kwargs)


def read_csv(path_or_buf, sep: str = r"\s+",
             comment: str = COMMENT_TOKEN,
             header: int | Sequence[int] | None | Literal["infer"] = "infer", *args, **kwargs):
    return pd.read_csv(path_or_buf, sep=sep, comment=comment, header=header, *args, **kwargs)


# Common Utilities ----------------------------------------------------------
def minimize_func(func, x_start: float, x_stop: float, ret_min_value: bool = False):
    """
    Minimizes the given function within (x_start, x_stop)

    if ret_min_value:
        returns the x that minimizes func and the value at minima as a tuple (min_x, func(x_minima))
    else:
        only returns x in range (x_start, x_stop) that minimizes func

    @param func: the function to be minimized, must take only a single-variable as input
    @param x_start: start of the domain range to search
    @param x_stop: start of the domain range to search
    @param ret_min_value: whether to return the value of function at x_minima i.e. func(x_minima)
    """
    opt_res = scipy.optimize.minimize_scalar(func, method="bounded", bounds=(x_start, x_stop))

    if ret_min_value:
        return opt_res.x, opt_res.fun
    return opt_res.x


def get_overlap_region(x1_start: float, x1_stop: float, x2_start: float, x2_stop: float):
    """
    @return: tuple representing the overlap range, or None if x1 and x2 do not overlap
    """
    start = max(x1_start, x2_start)
    stop = min(x1_stop, x2_stop)
    if start >= stop:
        return None

    return start, stop


def find_overlap_y_diff(x1: np.ndarray, y1: np.ndarray,
                        x2: np.ndarray, y2: np.ndarray):
    """
    @return the avg difference (y2 - y1) between the overlapping regions of x1 and x2
            if x1 and x2 do not overlap, returns None
            Assumes x1 and x2 are sorted in ascending order
    """
    # ALign overlapping regions of pmf1 and pmf2
    overlap = get_overlap_region(x1[0], x1[-1], x2[0], x2[-1])
    if overlap is None:
        return None    # No overlap

    start_x, stop_x = overlap

    start_x1_i = np.searchsorted(x1, start_x)
    start_x2_i = np.searchsorted(x2, start_x)
    stop_x1_i = np.searchsorted(x1, stop_x) - 1
    stop_x2_i = np.searchsorted(x2, stop_x) - 1

    len_x1 = (stop_x1_i - start_x1_i) + 1
    len_x2 = (stop_x2_i - start_x2_i) + 1
    common_len = min(len_x1, len_x2)

    return np.sum(y2[start_x2_i:start_x2_i + common_len] - y1[start_x1_i: start_x1_i + common_len]) / common_len


def load_df(file_path_or_buf,
            x_col_name: str,
            separator: str = r"\s+",
            x_start: float | None = None,
            x_end: float | None = None,
            sort_x: bool = False,
            drop_duplicates: bool = False,
            parsed_out_file_name: str | None = None,
            parsed_out_df_separator: str = "\t"):
    # Double Well
    df: pd.DataFrame = read_csv(file_path_or_buf, sep=separator)

    changed = False
    if sort_x:
        df.sort_values(x_col_name, inplace=True)
        changed = True

    if drop_duplicates:
        dup = df[df[x_col_name].duplicated()][x_col_name].unique()
        if len(dup) > 0:
            print(f"C.load_df => Duplicates found: {dup}")
            df.drop_duplicates([x_col_name], inplace=True)
            changed = True

    if changed and parsed_out_file_name:
        to_csv(df, parsed_out_file_name, sep=parsed_out_df_separator)
        print(f"C.load_df => Saving parsed DataFrame to file \"{parsed_out_file_name}\"")

    if x_start is not None:
        df = df[df[x_col_name] >= x_start]
    if x_end is not None:
        df = df[df[x_col_name] < x_end]

    return df
