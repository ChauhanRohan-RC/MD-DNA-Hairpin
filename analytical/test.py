import numpy as np
import matplotlib.pyplot as plt

from double_well_pmf import phi_scaled, double_well_pmf_scaled

Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)

T = 300  # Temperature (K)

A = -0.49
bias = 0.000236  # 0: symmetric, 0.0717: un-sym
ks = 10  # Force constant (kcal/mol/Å**2)

# Domain (in Å)
x = np.linspace(24, 42, 100, endpoint=False)  # [start = -1.5, stop = 1] with no scale or offset

pmf = double_well_pmf_scaled(x, depth=A, bias=bias, kb_t=Kb * T, ks=ks, x_offset=-32.467214, x_scale=0.198143, phi_offset=0.116308, phi_scale=0.364393)      # For un-symmetric potential

print("\n#X\tPMF")
for i in range(len(x)):
    print(f"{x[i]}\t{pmf[i]}")

plt.plot(x, pmf)
plt.show()