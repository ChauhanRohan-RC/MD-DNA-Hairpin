import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import C
from C import to_csv, COMMENT_TOKEN, PMF_FIT_COL_NAME_PARAM, PMF_FIT_COL_NAME_PARAM_VALUE, \
    PMF_FIT_COL_NAME_PARAM_STD_DEV, COL_NAME_X, COL_NAME_PMF_IMPOSED, minimize_func, COL_NAME_PMF
from double_well_pmf import double_well_pmf_scaled, phi_scaled

"""
Script to fit the double_well_pmf model to pmf samples
"""


def load_input_pmf(pmf_file_path_or_buf,
                   x_col_name: str,
                   pmf_col_name: str,
                   separator: str = r"\s+",
                   x_start: float | None = None,
                   x_end: float | None = None,
                   sort_x: bool = False,
                   drop_duplicates: bool = False,
                   return_meta_str: bool = False,
                   parsed_out_file_name: str | None = None,
                   parsed_out_df_separator: str = "\t"):
    # Double Well
    df = C.load_df(file_path_or_buf=pmf_file_path_or_buf,
                   x_col_name=x_col_name,
                   separator=separator,
                   x_start=x_start,
                   x_end=x_end,
                   sort_x=sort_x,
                   drop_duplicates=drop_duplicates,
                   parsed_out_file_name=parsed_out_file_name,
                   parsed_out_df_separator=parsed_out_df_separator)

    x = df[x_col_name].values
    pmf = df[pmf_col_name].values

    if return_meta_str:
        meta_str = f"INPUT pmf file: {pmf_file_path_or_buf} | x_column: {x_col_name} | pmf_column: {pmf_col_name}"
        return x, pmf, meta_str

    return x, pmf


def fit_double_well_pmf(x: np.ndarray, pmf: np.ndarray,
                        kb_t: float, ks: float,
                        out_file_name_prefix: str | None,
                        out_params_file_name: str | None = None,
                        out_covariances_file_name: str | None = None,
                        out_fit_pmf_samples_file_name: str | None = None,
                        out_fig_file_name: str | None = None,
                        fit_init_depth: float = -0.44,
                        fit_init_bias: float = 0,
                        fit_init_x_offset: float | None = None,
                        fit_init_x_scale: float | None = None,
                        fit_init_phi_offset: float | None = 0,
                        fit_init_phi_scale: float | None = 1,
                        interpolate_pmf: bool = True,
                        fit_interpolated_pmf: bool = False,
                        interpolate_sample_count: int = 200,
                        interpolate_kind: str = "quadratic",
                        interpolate_pmf_x_extra_left: float = 0,
                        interpolate_pmf_x_extra_right: float = 0,
                        meta_info_str: str | None = None,
                        out_fit_samples_x_col_name: str = COL_NAME_X,
                        out_fit_samples_pmf_col_name: str = COL_NAME_PMF):
    # Default initial offsets and scales for fitting
    if fit_init_x_offset is None:
        fit_init_x_offset = -(x[0] + x[-1]) / 2
    if fit_init_x_scale is None:
        fit_init_x_scale = abs(2 / (x[-1] - x[0]))
    if fit_init_phi_offset is None:
        fit_init_phi_offset = 0
    if fit_init_phi_scale is None:
        fit_init_phi_scale = 1

    # Interpolating Data
    if interpolate_sample_count < 1:
        interpolate_sample_count = 200  # Default
    pmf_interp_x = np.linspace(x[0] - interpolate_pmf_x_extra_left,
                               x[-1] + interpolate_pmf_x_extra_right,
                               num=interpolate_sample_count, endpoint=True)
    if interpolate_pmf:
        _pmf_interp_func = scipy.interpolate.interp1d(x, pmf,
                                                      kind=interpolate_kind,
                                                      fill_value="extrapolate")
        pmf_interp = _pmf_interp_func(pmf_interp_x)
    else:
        pmf_interp = None

    # Fitting just by data points instead of interpolated curve
    if interpolate_pmf and fit_interpolated_pmf:
        _fit_in_x = pmf_interp_x
        _fit_in_pmf = pmf_interp
    else:
        _fit_in_x = x
        _fit_in_pmf = pmf

        # Wrapper function used to fit the given data

    def _fit_func(_x: np.ndarray,
                  depth: float,
                  bias: float,
                  x_offset: float,
                  x_scale: float,
                  phi_offset: float,
                  phi_scale: float) -> np.ndarray:

        return double_well_pmf_scaled(x=_x,
                                      kb_t=kb_t,
                                      ks=ks,
                                      depth=depth,
                                      bias=bias,
                                      x_offset=x_offset,
                                      x_scale=x_scale,
                                      phi_offset=phi_offset,
                                      phi_scale=phi_scale)

    param_opt_vals, param_covariances = scipy.optimize.curve_fit(_fit_func,
                                                                 xdata=_fit_in_x,
                                                                 ydata=_fit_in_pmf,
                                                                 p0=(fit_init_depth, fit_init_bias,
                                                                     fit_init_x_offset, fit_init_x_scale,
                                                                     fit_init_phi_offset, fit_init_phi_scale),
                                                                 # bounds=([-0.5, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                                                                 #         [1e-4, np.inf, np.inf, np.inf, np.inf, np.inf])
                                                                 )

    param_names = ["depth", "bias", "x_offset", "x_scale", "phi_offset", "phi_scale"]
    # param_val_str = "\n".join(zip(param_names, param_opt_vals))

    param_std_dev = np.sqrt(np.diag(param_covariances))
    # param_std_dev_str = "\n".join(zip(param_names, param_std_dev))

    param_df: pd.DataFrame = pd.DataFrame({
        PMF_FIT_COL_NAME_PARAM: param_names,
        PMF_FIT_COL_NAME_PARAM_VALUE: param_opt_vals,
        PMF_FIT_COL_NAME_PARAM_STD_DEV: param_std_dev
    })

    print("------------- FIT PARAMETERS ------------")
    print(param_df)
    print("-----------------------------------------")

    if out_file_name_prefix:
        if not out_params_file_name:
            out_params_file_name = f"{out_file_name_prefix}.params.txt"
        if not out_covariances_file_name:
            out_covariances_file_name = f"{out_file_name_prefix}.covariances.txt"
        if not out_fit_pmf_samples_file_name:
            out_fit_pmf_samples_file_name = f"{out_file_name_prefix}.fit_samples.csv"
        if not out_fig_file_name:
            out_fig_file_name = f"{out_file_name_prefix}.plot.pdf"

    meta_info = (f"{COMMENT_TOKEN} INPUT Thermal Energy (KbT): {kb_t}\n"
                 f"{COMMENT_TOKEN} INPUT Spring constant (Ks): {ks}\n")

    if meta_info_str:
        meta_info = f"{COMMENT_TOKEN} {meta_info_str}\n" + meta_info

    # Writing output
    if out_params_file_name:
        with open(out_params_file_name, "w") as out_p:
            out_p.write(f"{COMMENT_TOKEN} -------------- OPTIMIZED DOUBLE-WELL PMF PARAMETERS ----------------\n")
            out_p.write(meta_info)
            out_p.write(f"{COMMENT_TOKEN} ----------------------------------------\n")
            to_csv(param_df, out_p, mode="a")

        print(f"PMF_FIT: Parameters saved to file \"{out_params_file_name}\"")

    if out_covariances_file_name:
        np.savetxt(out_covariances_file_name, param_covariances)
        print(f"PMF_FIT: Covariances saved to file \"{out_covariances_file_name}\"")

    # Sampling fit function
    fit_pmf = _fit_func(pmf_interp_x, *param_opt_vals)
    if out_fit_pmf_samples_file_name:
        fit_pmf_df = pd.DataFrame({
            out_fit_samples_x_col_name: pmf_interp_x,
            out_fit_samples_pmf_col_name: fit_pmf
        })

        with open(out_fit_pmf_samples_file_name, "w") as out_p:
            out_p.write(f"{COMMENT_TOKEN} -------------- DOUBLE-WELL PMF FIT SAMPLES ----------------\n")
            out_p.write(meta_info)
            param_val_str = " | ".join(f"{k}: {v}" for k, v in zip(param_names, param_opt_vals))
            out_p.write(f"{COMMENT_TOKEN} Double-Well PMF FIT Parameters => {param_val_str}\n")
            out_p.write(f"{COMMENT_TOKEN} ----------------------------------------\n")
            to_csv(fit_pmf_df, out_p, mode="a")

        print(f"PMF_FIT: PMF-FIT Samples saved to file \"{out_fit_pmf_samples_file_name}\"")

    # Plot
    plt.rcParams["figure.figsize"] = (12, 9)  # Figure Size (in inches)
    plt.scatter(x, pmf, label="Input PMF")
    if interpolate_pmf:
        plt.plot(pmf_interp_x, pmf_interp, "--", label="")
    plt.plot(pmf_interp_x, fit_pmf, label="Double-Well FIT")

    # TEST
    # param_opt_vals[1] = 0
    # fit_pmf2 = _fit_func(pmf_interp_x, *param_opt_vals)
    # plt.plot(pmf_interp_x, fit_pmf2, label="FIT-2")

    plt.legend(loc="upper right")
    if out_fig_file_name:
        plt.savefig(out_fig_file_name)
        print(f"PMF_FIT: Plot saved to file \"{out_fig_file_name}\"")
    plt.show()


def load_fit_params(fit_param_file) -> np.ndarray:
    _param_df: pd.DataFrame = pd.read_csv(fit_param_file, comment=COMMENT_TOKEN, sep=r"\s+")
    return _param_df[PMF_FIT_COL_NAME_PARAM_VALUE].values


def load_double_well_phi_func(fit_param_file, kb_t: float, ks: float):
    _fit_params = load_fit_params(fit_param_file)
    return lambda x: phi_scaled(x, kb_t, ks, *_fit_params)


def load_double_well_pmf_func(fit_param_file, kb_t: float, ks: float):
    _fit_params = load_fit_params(fit_param_file)
    return lambda x: double_well_pmf_scaled(x, kb_t, ks, *_fit_params)


def samplify_double_well_pmf_fit(fit_param_file, kb_t: float, ks: float,
                                 x_start: float, x_stop: float,
                                 sample_count: int, output_sample_file: str):
    func = load_double_well_pmf_func(fit_param_file, kb_t=kb_t, ks=ks)

    x = np.linspace(x_start, x_stop, sample_count, endpoint=True)
    y = func(x)

    df = pd.DataFrame({
        COL_NAME_X: x,
        COL_NAME_PMF_IMPOSED: y
    })

    df.to_csv(output_sample_file, sep="\t", header=True, index=False, index_label=False)

    plt.plot(x, y)
    plt.show()


# Returns the minima coordinates (x, y) where x is in between (x_start, x_stop)
def minimize_double_well_pmf(fit_param_file, kb_t: float, ks: float,
                             x_start: float, x_stop: float, ret_min_value: bool = False):
    pmf_func = load_double_well_pmf_func(fit_param_file, kb_t=kb_t, ks=ks)

    return minimize_func(pmf_func, x_start=x_start, x_stop=x_stop, ret_min_value=ret_min_value)


if __name__ == '__main__':
    if 0:  # Find minima of fitted double-well
        min_val = minimize_double_well_pmf("results-sim_double_well_fit/fit_params-1.2.txt",
                                           1.9872036e-3 * 300, 10,
                                           13, 16, ret_min_value=True)

        print(min_val)

    if 0:  # Samplify double-well pmf
        samplify_double_well_pmf_fit("results-sim_double_well_fit/fit_params-1.2.txt",
                                     1.9872036e-3 * 300, 10,
                                     13.68, 25.45, 500,
                                     "results-sim_double_well_fit/fit_samples-1.2.csv")

    # ---------------------------------------------------------------
    ## NOTE: Double well Minima(s)
    # fit_params-1.1.txt -> 14.963448853268662, 24.210883674081007
    # fit_params-1.2.txt -> 14.988497234334494, 24.25070294049196
    # fit_params-2.1.txt -> 26.887663450991564, 38.00000736802938
    # fit_params-2.2.txt -> 26.637210420334856, 38.45796438483576
    pass
