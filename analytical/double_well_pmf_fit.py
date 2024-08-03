import math
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from double_well_pmf import double_well_pmf_scaled

comment_token = "#"

# Constants ----------------------------------------------
Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol

Ks = 10  # Force constant (kcal/mol/Å**2)


# Wrapper function used to fit the given data
def double_well_pmf_fit_func(x: np.ndarray,
                             depth: float,
                             bias: float,
                             x_offset: float,
                             x_scale: float,
                             phi_offset: float,
                             phi_scale: float) -> np.ndarray:
    return double_well_pmf_scaled(x, depth, bias, KbT, Ks, x_offset, x_scale, phi_offset, phi_scale)


# INPUT -------------------------------------------------
data_file_name = "sp_traj1.csv"
data_x_col_name = "EXT_BIN_MED"
data_pmf_col_name = "PMF_RE"

data_x_start = None         # (optional) Inclusive
data_x_end = None           # (optional) Exclusive

interpolate_data = True             # Interpolate raw data for smoothness
interpolate_sample_count = 100
fit_interpolated_data = False       # Use interpolated data for Double-Well fit. [Only if interpolate_data is True]

fit_init_depth = -0.44      # Initial DEPTH param for Double-Well fit
fit_init_bias = 0.01        # Initial BIAS param for Double-Well fit

# OUTPUT -----------------------------------------------
output_params_file = "fit_params-1.txt"         # (optional) Optimized Fit params
output_covar_file = "fit_covariances-1.txt"     # (optional) Covariance Matrix of Fit-params
output_fig_file = "fit-1.svg"                   # (optional) save fit plot

# -----------------------------------------------------------------------------------

# Double Well
data_df = pd.read_csv(data_file_name, comment=comment_token, sep=r"\s+")
# db_well = pd.read_csv("sp_traj2.csv", comment="#", delimiter=r"\s+")

if data_x_start is not None:
    data_df = data_df[data_df[data_x_col_name] >= data_x_start]

if data_x_end is not None:
    data_df = data_df[data_df[data_x_col_name] < data_x_end]

data_x = data_df[data_x_col_name].values
data_pmf = data_df[data_pmf_col_name].values

# Interpolating Data
if interpolate_sample_count < 1:
    interpolate_sample_count = 100      # default

data_interp_x = np.linspace(data_x[0] - 1, data_x[-1] + 1, interpolate_sample_count)
if interpolate_data:
    _data_interp_func = sp.interpolate.interp1d(data_x, data_pmf, kind="quadratic", fill_value="extrapolate")
    data_interp_pmf = _data_interp_func(data_interp_x)
else:
    data_interp_pmf = None

# Initial Offsets and scales for fitting
fit_init_x_offset = -(data_x[0] + data_x[-1]) / 2
fit_init_x_scale = abs(2 / (data_x[-1] - data_x[0]))
fit_init_phi_offset = 0
fit_init_phi_scale = 1

# Fitting just by data points instead of interpolated curve
if interpolate_data and fit_interpolated_data:
    _fit_in_x = data_interp_x
    _fit_in_pmf = data_interp_pmf
else:
    _fit_in_x = data_x
    _fit_in_pmf = data_pmf

param_opt_vals, param_covars = curve_fit(double_well_pmf_fit_func,
                                         xdata=_fit_in_x,
                                         ydata=_fit_in_pmf,
                                         p0=(fit_init_depth, fit_init_bias, fit_init_x_offset, fit_init_x_scale, fit_init_phi_offset, fit_init_phi_scale))

param_names = ["depth", "bias", "x_offset", "x_scale", "phi_offset", "phi_scale"]
# param_val_str = "\n".join(zip(param_names, param_opt_vals))

param_std_dev = np.sqrt(np.diag(param_covars))
# param_std_dev_str = "\n".join(zip(param_names, param_std_dev))

param_df: pd.DataFrame = pd.DataFrame({
    "PARAM": param_names,
    "VALUE": param_opt_vals,
    "STD_DEV": param_std_dev
})

print("------------- FIT PARAMETERS ------------")
print(param_df)
print("-----------------------------------------")

# Writing output
if output_params_file:
    with open(output_params_file, "w") as out_p:
        out_p.write(f"{comment_token} -------------- OPTIMIZED DOUBLE-WELL PMF PARAMETERS ----------------\n")
        out_p.write(f"{comment_token} INPUT data file: {data_file_name}  |  x_column: {data_x_col_name}  |  pmf_column: {data_pmf_col_name}\n")
        out_p.write(f"{comment_token} INPUT Thermal Energy (KbT): {KbT}\n")
        out_p.write(f"{comment_token} INPUT Spring constant (Ks): {Ks}\n")
        out_p.write(f"{comment_token} ---------------------------------------\n")

        param_df.to_csv(out_p, mode="a", sep="\t", header=True, index=False, index_label=False)

if output_covar_file:
    np.savetxt(output_covar_file, param_covars)

# Sampling fit function
fit_pmf = double_well_pmf_fit_func(data_interp_x, *param_opt_vals)

# Plot
plt.rcParams["figure.figsize"] = (12, 9)     # Figure Size (in inches)
plt.scatter(data_x, data_pmf)
if interpolate_data:
    plt.plot(data_interp_x, data_interp_pmf, "--", label="Interpolated")
plt.plot(data_interp_x, fit_pmf, label="Double-Well FIT")

plt.legend(loc="upper right")
if output_fig_file:
    plt.savefig(output_fig_file)

plt.show()

# FIT PARAMS: depth=--0.496145547, bias=0.00393961445, x_offset=-19.9718998, x_scale=0.201206774, phi_offset=0.0318009327, phi_scale=0.482772851
