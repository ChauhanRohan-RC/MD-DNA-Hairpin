import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from double_well_pmf import double_well_pmf_scaled, phi_scaled, minimize_func

"""
Script to fit the double_well_pmf model to pmf samples

NOTE: Search TODO and set required params
"""

comment_token = "#"

COL_NAME_X = "X"
COL_NAME_PMF_IMPOSED = "PMF_IM"

FIT_COL_NAME_PARAM = "PARAM"
FIT_COL_NAME_PARAM_VALUE = "VALUE"
FIT_COL_NAME_STD_DEV = "STD_DEV"

# Constants: TODO ----------------------------------------------
Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol

Ks = 10  # Force constant (kcal/mol/Å**2)

# INPUT: TODO -------------------------------------------------
pmf_data__file_name = "sp_traj1.2.csv"
pmf_data__x_col_name = "EXT_BIN_MED"
pmf_data__pmf_col_name = "PMF_RE"

pmf_data_x_start = None  # (optional) Inclusive
pmf_data_x_end = None  # (optional) Exclusive

interpolate_pmf = True  # Interpolate raw data for smoothness
interpolate_sample_count = 200
interpolate_kind = "quadratic"  # see {@link scipy.interpolate.interp1d}
interpolate_pmf_x_extra_left = 2  # (optional) extra domain to extrapolate the input pmf samples
interpolate_pmf_x_extra_right = 1  # (optional) extra domain to extrapolate the input pmf samples

fit_interpolated_pmf = False  # Use interpolated data for Double-Well fit. [Only if interpolate_data is True]


fit_init_depth = -0.44  # Initial DEPTH param for Double-Well fit
fit_init_bias = 0.01  # Initial BIAS param for Double-Well fit

# OUTPUT: TODO  -----------------------------------------------
out_fit_params_file = "results-double_well_fit/fit_params-1.2.txt"  # (optional) Optimized Fit params
out_fit_covar_file = "results-double_well_fit/fit_covariances-1.2.txt"  # (optional) Covariance Matrix of Fit-params
out_fit_fig_file = "results-double_well_fit/fit-1.2.pdf"  # (optional) save fit plot


# -----------------------------------------------------------------------------------


def fit_double_well_pmf():
    # Double Well
    pmf_data_df = pd.read_csv(pmf_data__file_name, comment=comment_token, sep=r"\s+")
    # db_well = pd.read_csv("sp_traj2.csv", comment="#", delimiter=r"\s+")

    if pmf_data_x_start is not None:
        pmf_data_df = pmf_data_df[pmf_data_df[pmf_data__x_col_name] >= pmf_data_x_start]

    if pmf_data_x_end is not None:
        pmf_data_df = pmf_data_df[pmf_data_df[pmf_data__x_col_name] < pmf_data_x_end]

    in_pmf_x = pmf_data_df[pmf_data__x_col_name].values
    in_pmf = pmf_data_df[pmf_data__pmf_col_name].values

    # Initial Offsets and scales for fitting: TODO: initial fit offsets and scales (leave as default for most cases)
    fit_init_x_offset = -(in_pmf_x[0] + in_pmf_x[-1]) / 2
    fit_init_x_scale = abs(2 / (in_pmf_x[-1] - in_pmf_x[0]))
    fit_init_phi_offset = 0
    fit_init_phi_scale = 1

    # Interpolating Data
    _interp_samples = interpolate_sample_count if interpolate_sample_count > 0 else 100  # DEFAULT
    pmf_interp_x = np.linspace(in_pmf_x[0] - interpolate_pmf_x_extra_left,
                               in_pmf_x[-1] + interpolate_pmf_x_extra_right,
                               num=_interp_samples, endpoint=True)
    if interpolate_pmf:
        _pmf_interp_func = scipy.interpolate.interp1d(in_pmf_x, in_pmf,
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
        _fit_in_x = in_pmf_x
        _fit_in_pmf = in_pmf

        # Wrapper function used to fit the given data

    def __double_well_pmf_fit_func(x: np.ndarray,
                                   depth: float,
                                   bias: float,
                                   x_offset: float,
                                   x_scale: float,
                                   phi_offset: float,
                                   phi_scale: float) -> np.ndarray:

        return double_well_pmf_scaled(x=x,
                                      kb_t=Kb * T,
                                      ks=Ks,
                                      depth=depth,
                                      bias=bias,
                                      x_offset=x_offset,
                                      x_scale=x_scale,
                                      phi_offset=phi_offset,
                                      phi_scale=phi_scale)

    param_opt_vals, param_covars = scipy.optimize.curve_fit(__double_well_pmf_fit_func,
                                                            xdata=_fit_in_x,
                                                            ydata=_fit_in_pmf,
                                                            p0=(fit_init_depth, fit_init_bias, fit_init_x_offset,
                                                                fit_init_x_scale,
                                                                fit_init_phi_offset, fit_init_phi_scale))

    param_names = ["depth", "bias", "x_offset", "x_scale", "phi_offset", "phi_scale"]
    # param_val_str = "\n".join(zip(param_names, param_opt_vals))

    param_std_dev = np.sqrt(np.diag(param_covars))
    # param_std_dev_str = "\n".join(zip(param_names, param_std_dev))

    param_df: pd.DataFrame = pd.DataFrame({
        FIT_COL_NAME_PARAM: param_names,
        FIT_COL_NAME_PARAM_VALUE: param_opt_vals,
        FIT_COL_NAME_STD_DEV: param_std_dev
    })

    print("------------- FIT PARAMETERS ------------")
    print(param_df)
    print("-----------------------------------------")

    # Writing output
    if out_fit_params_file:
        with open(out_fit_params_file, "w") as out_p:
            out_p.write(f"{comment_token} -------------- OPTIMIZED DOUBLE-WELL PMF PARAMETERS ----------------\n")
            out_p.write(
                f"{comment_token} INPUT data file: {pmf_data__file_name}  |  x_column: {pmf_data__x_col_name}  |  pmf_column: {pmf_data__pmf_col_name}\n")
            out_p.write(f"{comment_token} INPUT Thermal Energy (KbT): {KbT}\n")
            out_p.write(f"{comment_token} INPUT Spring constant (Ks): {Ks}\n")
            out_p.write(f"{comment_token} ---------------------------------------\n")

            param_df.to_csv(out_p, mode="a", sep="\t", header=True, index=False, index_label=False)

    if out_fit_covar_file:
        np.savetxt(out_fit_covar_file, param_covars)

    # Sampling fit function
    fit_pmf = __double_well_pmf_fit_func(pmf_interp_x, *param_opt_vals)

    # Plot
    plt.rcParams["figure.figsize"] = (12, 9)  # Figure Size (in inches)
    plt.scatter(in_pmf_x, in_pmf)
    if interpolate_pmf:
        plt.plot(pmf_interp_x, pmf_interp, "--", label="Interpolated")
    plt.plot(pmf_interp_x, fit_pmf, label="Double-Well FIT")

    plt.legend(loc="upper right")
    if out_fit_fig_file:
        plt.savefig(out_fit_fig_file)

    plt.show()


def load_fit_params(fit_param_file) -> np.ndarray:
    _param_df: pd.DataFrame = pd.read_csv(fit_param_file, comment=comment_token, sep=r"\s+")
    return _param_df[FIT_COL_NAME_PARAM_VALUE].values


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
                             x_start: float, x_stop: float):
    pmf_func = load_double_well_pmf_func(fit_param_file, kb_t=kb_t, ks=ks)

    return minimize_func(pmf_func, x_start=x_start, x_stop=x_stop)


if __name__ == '__main__':
    if 0:  # Fit double well pmf
        fit_double_well_pmf()

    if 0:  # Find minima of fitted double-well
        min_val = minimize_double_well_pmf("results-double_well_fit/fit_params-1.2.txt",
                                           1.9872036e-3 * 300, 10,
                                           13, 16)

        print(min_val)

    if 0:  # Samplify double-well pmf
        samplify_double_well_pmf_fit("results-double_well_fit/fit_params-1.2.txt",
                                     1.9872036e-3 * 300, 10,
                                     13.68, 25.45, 500,
                                     "results-double_well_fit/fit_samples-1.2.csv")

    # ---------------------------------------------------------------
    ## NOTE: Double well Minima(s)
    # fit_params-1.1.txt -> 14.963448853268662, 24.210883674081007
    # fit_params-1.2.txt -> 14.988497234334494, 24.25070294049196
    # fit_params-2.1.txt -> 26.887663450991564, 38.00000736802938
    # fit_params-2.2.txt -> 26.637210420334856, 38.45796438483576
    pass
