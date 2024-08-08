import numpy as np
import scipy as scipy
import pandas as pd
import math
import matplotlib.pyplot as plt

from double_well_pmf import phi, phi_scaled, double_well_pmf_scaled, double_well_pmf
from double_well_pmf_fit import load_fit_params

"""
Implementation of the FINAL EQUATION of Sp(c)
"""

def dn(x: np.ndarray | float, n: int, a: float, hermite_monic=True) -> np.ndarray:
    _hermite_poly = scipy.special.hermite(n, monic=hermite_monic)
    return math.exp2(-n / 2) * np.exp(-a * np.square(x) / 2) * _hermite_poly(x * math.sqrt(a))


def kn(n: int, depth: float, ks: float, friction_coeff: float) -> float:
    return (n + depth + 0.5) * ks / friction_coeff


## CONSTANTS -----------------------------
Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol

Ks = 10  # Force constant (kcal/mol/Å**2)

# PARAMS
x_a = 13.3     # LEFT Boundary (Å)
x_b = 25.5     # RIGHT Boundary (Å)

cyl_dn_a = 10          # "a" param of cylindrical function
friction_coeff = 1e-7   # friction coeff (eta_1) (in kcal.sec/mol/Å**2). In range (0.5 - 2.38) x 10-7
d1 = KbT / friction_coeff   # diffusion coefficient with beta=1     (in Å**2/s)

fit_params_file = "fit_params-1.txt"
pmf_fit_params = load_fit_params(fit_params_file)
depth, bias, x_offset, x_scale, phi_offset, phi_scale = pmf_fit_params


# Internal Phi(A, x)
# def phi(x: np.ndarray | float):
#     return phi(x=x, kb_t=KbT, ks=Ks,
#                       depth=depth, bias=bias)


def phi_sc(x: np.ndarray | float):
    return phi_scaled(x=x, kb_t=KbT, ks=Ks,
                      depth=depth, bias=bias,
                      x_offset=0, x_scale=1,
                      phi_offset=0, phi_scale=1)


def dn_by_phi(x: np.ndarray| float, n: int, dn_a: float):
    return dn(x, n=n, a=dn_a) / phi_sc(x)


def dn_by_phi_func(n: int, dn_a: float):
    return lambda x: dn_by_phi(x, n=n, dn_a=dn_a)


def sp(x_ab: float, x_a: float = x_a, x_b: float = x_b, cyl_dn_a: float = cyl_dn_a, n_max: int = 100) -> float:
    _pre_c: float = d1 * math.sqrt(Ks / (KbT * 2 * math.pi))
    _phi_x = phi_sc(x_ab)

    _pre = _pre_c * _phi_x * _phi_x

    def _cal_summand(n: int, inv_fact: float) -> float:
        __dn_by_phi_func = dn_by_phi_func(n, cyl_dn_a)

        __der = scipy.misc.derivative(__dn_by_phi_func, x0=x_ab, dx=1e-4)
        __c = __dn_by_phi_func(x_b) - __dn_by_phi_func(x_a)
        __integral = 1 / kn(n=n, depth=depth, ks=Ks, friction_coeff=friction_coeff)

        return inv_fact * __der * __c * __integral

    _sum: float = 0.0

    # first term (n = 0)
    _sum += _cal_summand(0, 1)

    # Rest terms (n > 1)
    inv_factorial: float = 1.0
    for n in range(1, n_max):
        inv_factorial /= n
        _sum += _cal_summand(n, inv_factorial)

    return _pre * _sum



def test():

    x_a, x_b = -1.05, 0.95
    n = 0
    n_max = 10
    cyl_dn_a = 1

    # x = np.linspace(x_a, x_b, 100, endpoint=True)
    # _x = (x + x_offset) * x_scale

    # y = np.zeros(len(_x))
    #
    # for i in range(len(_x)):
    #     y[i] = sp(_x[i], x_a=(x_a + x_offset) * x_scale, x_b=(x_b + x_offset) * x_scale, n_max=100)


    x = np.linspace(x_a, x_b, 100, endpoint=True)

    # y1 = 1/ np.square(phi_sc(x))
    # y1 = double_well_pmf(x, KbT, Ks, depth, bias)
    # y2 = dn(x, n, cyl_dn_a, True)
    # y = dn_by_phi(x, n, cyl_dn_a)
    # y = d1 * math.sqrt(Ks / (KbT * 2 * math.pi)) * (dn_by_phi(0.2, 0, cyl_dn_a) - dn_by_phi(-0.2, 0, cyl_dn_a)) * (1 / kn(0, depth, Ks, friction_coeff)) * np.square(phi_sc(x)) * np.gradient(dn_by_phi(x, 0, cyl_dn_a), x)



    # plt.plot(x, y1, label="PMF-IM")
    # plt.plot(x, y2, label="Dn")
    # # plt.plot(x, y2/y1, label="Dn/Phi")
    #
    # y3 = np.gradient(y2 / y1, x)
    # plt.plot(x, y3, label="der(Dn/Phi)")
    # plt.plot(x, y3 * y1 * y1, label="Sp-1")

    y5 = np.zeros(len(x))

    for i in range(len(x)):
        y5[i] = sp(x[i], x_a=x_a, x_b=x_b, cyl_dn_a=cyl_dn_a, n_max=n_max)

    y5 -= y5.min()

    plt.plot(x, y5, label="SP")
    #
    # print(f"y5 MIN: {y5.min()}, MAX: {y5.max()}")

    # # Integral in the denominator = Constant
    # c = scipy.integrate.trapezoid(y=y5, x=x)
    #
    # y6 = np.zeros(len(x), dtype=np.float128)
    #
    # for i in range(len(x)):
    #     _v = scipy.integrate.trapezoid(y=y5[i:], x=x[i:])
    #     y6[i] = _v / c

    # plt.plot(x, y6, label="SP")

    # grad: np.ndarray = np.gradient(y6, x)
    # print(f"IS gradient +ve: {(grad > 0).sum()}")
    #
    # # y7 = -grad
    # y7 = KbT * np.log(-grad)
    # plt.plot(x, y7, label="PMF-RE")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    test()
