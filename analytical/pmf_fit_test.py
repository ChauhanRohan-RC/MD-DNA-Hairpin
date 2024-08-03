import math
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from double_well_pmf import double_well_pmf_scaled, phi

Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol

Ks = 10  # Force constant (kcal/mol/Å**2)


def pmf_fit_func1(x: np.ndarray, depth: float, bias: float,
                  a: float, b: float, c: float, d: float):
    x = (a * x) + b
    return (c * double_well_pmf_scaled(x=x,
                                       depth=depth,
                                       bias=bias,
                                       kb_t=KbT,
                                       ks=Ks,
                                       x_scale=1,
                                       x_offset=0,
                                       phi_scale=1,
                                       phi_offset=0)) + d


def pmf_fit_func2(x: np.ndarray, depth: float, bias: float,
                  a: float, b: float, phi_scale: float, phi_scaled_offset: float):
    x = (a * x) + b

    _phi = phi(x=x,
               depth=depth,
               bias=bias,
               kb_t=KbT,
               ks=Ks)

    _phi_scaled = (_phi * phi_scale) + phi_scaled_offset
    return 2 * KbT * np.log(_phi_scaled)


# df = pd.read_csv("sp_traj_pmf_merged.csv", comment="#", delimiter=r"\s+")
#
# well1 = df[df["EXT_BIN"] <= 17]
# well2 = df[df["EXT_BIN"] >= 12]

well1 = pd.read_csv("sp_traj1.csv", comment="#", delimiter=r"\s+")
well2 = pd.read_csv("sp_traj2.csv", comment="#", delimiter=r"\s+")

# well1_init_depth = -0.44
# well1_init_bias = 0.05
# well1_init_x_scale = 1
# well1_init_x_scaled_offset = 0
# well1_init_phi_scale = 1
# well1_init_phi_scaled_offset = 0

# popt, pcov = curve_fit(f=pmf_fit_func,
#                        xdata=well1["EXT_BIN_MED"].values,
#                        ydata=well1["PMF_RE"].values, p0=(0.04, 1/8, -2.7, 1, 0))
#
# print(f"OPTIMIZED PARAMS: {popt}")
# print("\n")
# print(pcov)
# print("\n")
# print(f"PARAMS STD DEV: {np.sqrt(np.diag(pcov))}")


# ----------------------------------------------------------------------------------
# Well 1 Optimized FIT: depth -0.44, bias 0.02  , a 1/7.5, b -2.7, c 2, d -0.8
# Well 1 Optimized FIT MAIN FUN: -0.49349084  0.00547092  0.17657133 -3.60012839  0.48751877  0.01932369

xfit = np.linspace(12.5, 27.5, 50)
yfit1 = pmf_fit_func1(xfit, -0.44, 0.02, 1 / 7.5, -2.7, 2, -0.8)

# popt, pcov = curve_fit(f=pmf_fit_func_main,
#           xdata=xfit,
#           ydata=yfit,
#           p0=(-0.44, 0.02  , 1/7.5, -2.7, 1, 0))
#
# print(f"OPTIMIZED PARAMS: {popt}")
# print("\n")
# print(pcov)
# print("\n")
# print(f"PARAMS STD DEV: {np.sqrt(np.diag(pcov))}")

yfit2 = pmf_fit_func2(xfit,
                      -0.49349084,
                      0.00547092,
                      0.1765713,
                      -3.60012839,
                      0.48751877,
                      0.01932369)

plt.scatter(well1["EXT_BIN_MED"], well1["PMF_RE"])
plt.plot(xfit, yfit2)
plt.show()
# ----------------------------------------------------------------------------------
