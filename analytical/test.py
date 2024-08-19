import numpy as np

from C import BOLTZMANN_CONST_KCAL_PER_MOL_K
from double_well_pmf import double_well_pmf_scaled
from double_well_pmf_fit import load_fit_params

import matplotlib.pyplot as plt

T = 300  # Temperature (K)
KbT = BOLTZMANN_CONST_KCAL_PER_MOL_K * T  # (kcal/mol)
Ks = 10  # Force constant (kcal/mol/Å**2)

params = load_fit_params("data_exp/pmf_fit/pmf-1.1.params.txt")
# params[0] = -0.44
params[1] = 0.0

x = np.linspace(-130, 120, 50)
y = double_well_pmf_scaled(x, KbT, Ks, *params)

plt.plot(x, y)
plt.show()