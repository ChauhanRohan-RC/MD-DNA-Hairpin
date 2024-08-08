import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

df = pd.read_csv("sp_traj2.csv", comment="#", sep=r"\s+")

data_x = df["EXT_BIN_MED"].values
data_y = df["SP"].values

interp_x = np.linspace(data_x[0], data_x[-1], 100)
interp_func = sp.interpolate.interp1d(data_x, data_y, kind="quadratic", fill_value="extrapolate")
interp_y = interp_func(interp_x)


def sigmoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (a / (1 + np.exp(b * (x + c)))) + d


parms, cov = sp.optimize.curve_fit(sigmoid, data_x, data_y, p0=(1, 1, -30, 0))
print(parms)

fit_func = lambda x: sigmoid(x, *parms)
fit_y = fit_func(interp_x)

# diff = np.zeros(len(interp_x))
# for i in range(len(interp_x)):
#     diff[i] = sp.misc.derivative(fit_func, interp_x[i], dx=1e-4)

plt.scatter(data_x, data_y)
plt.plot(interp_x, interp_y, "--")
plt.plot(interp_x, fit_y)
# plt.plot(interp_x, -diff)
plt.show()

## FIT PARAMS
#           a           b           c               d
# SP 1: 0.9647104    1.49742155 -19.83571486   0.03906433
# SP 2: 0.97349659   0.92763962 -32.36484571   0.03530707
