import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy

from double_well_pmf import phi_scaled

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
CORE IMPLEMENTATION of Splitting Probability from the ground up

1. conditional probability:  P(x,t | x0,t0)
2. conditional probability integral w.r.t to x at given t, x0, t0
3. First Passage Time Distribution:  FPT(x0, t)
4. Splitting probability from FPT:  Sp(x0) = time_integral(FPT(x0, t))


NOTE: functions ending with "vec" are vectorized versions of corresponding functions.
      They accept array of inputs, and compute the base function at each input in a for-loop
"""


def mittag_leffler(x: int | float, a: int | float, b: int | float, k_max: int = 50) -> np.float128:
    _sum = np.float128(0)

    _x_pow = np.float128(1)
    for k in range(0, k_max):
        _val = _x_pow / scipy.special.gamma((a * k) + b)

        _sum += _val
        _x_pow *= x

    return _sum


def mittag_leffler_vec(x: np.ndarray | int | float,
                       a: np.ndarray | int | float,
                       b: np.ndarray | int | float,
                       k_max: np.ndarray | int = 50) -> np.ndarray | np.float128:
    _vec = np.vectorize(mittag_leffler, otypes=[np.float128])
    return _vec(x=x, a=a, b=b, k_max=k_max)


def dn(x: np.ndarray | float, n: int, a: float, hermite_monic=True) -> np.ndarray | float:
    _hermite_poly = scipy.special.hermite(n, monic=hermite_monic)
    return math.exp2(-n / 2) * np.exp(-a * np.square(x) / 2) * _hermite_poly(x * math.sqrt(a))


def kn(n: int, depth: float, ks: float, friction_coeff: float) -> float:
    return (n + depth + 0.5) * ks / friction_coeff


# Internal Phi(A, x)
# def phi(x: np.ndarray | float):
#     return phi(x=x, kb_t=KbT, ks=Ks,
#                       depth=depth, bias=bias)


# def phi_sc(x: np.ndarray | float,
#            kb_t: float,
#            ks: float,
#            depth: float,
#            bias: float,
#            x_offset: float = 0,
#            x_scale: float = 1,
#            phi_offset: float = 0,
#            phi_scale: float = 1) -> np.ndarray | float:
#     return phi_scaled(x=x, kb_t=kb_t, ks=ks,
#                       depth=depth, bias=bias,
#                       x_offset=x_offset, x_scale=x_scale,
#                       phi_offset=phi_offset, phi_scale=phi_scale)


def dn_by_phi(x: np.ndarray | float, n: int, dn_a: float,
              kb_t: float,
              ks: float,
              depth: float,
              bias: float,
              x_offset: float = 0,
              x_scale: float = 1,
              phi_offset: float = 0,
              phi_scale: float = 1) -> np.ndarray | float:
    return dn(x, n=n, a=dn_a) / phi_scaled(x, kb_t=kb_t, ks=ks,
                                           depth=depth, bias=bias,
                                           x_offset=x_offset, x_scale=x_scale,
                                           phi_offset=phi_offset, phi_scale=phi_scale)


def dn_by_phi_func(n: int, dn_a: float,
                   kb_t: float,
                   ks: float,
                   depth: float,
                   bias: float,
                   x_offset: float = 0,
                   x_scale: float = 1,
                   phi_offset: float = 0,
                   phi_scale: float = 1):
    return lambda x: dn_by_phi(x, n=n, dn_a=dn_a,
                               kb_t=kb_t, ks=ks,
                               depth=depth, bias=bias,
                               x_offset=x_offset, x_scale=x_scale,
                               phi_offset=phi_offset, phi_scale=phi_scale)


def cond_prob(x: float, t: float, x0: float, t0: float,
              n_max: int,
              cyl_dn_a: float,
              kb_t: float,
              ks: float,
              friction_coeff: float,
              depth: float,
              bias: float,
              x_offset: float = 0,
              x_scale: float = 1,
              phi_offset: float = 0,
              phi_scale: float = 1) -> np.float128:
    bias_crit = math.sqrt(2) * scipy.special.gamma((depth / 2) + 0.75) / scipy.special.gamma((depth / 2) + 0.25)
    c0_sq = ((bias_crit ** 2) - (bias ** 2)) / (2 * bias_crit)

    # d1 = kb_t / friction_coeff

    def _phi(_x):
        return phi_scaled(_x, kb_t=kb_t, ks=ks,
                          depth=depth, bias=bias,
                          x_offset=x_offset, x_scale=x_scale,
                          phi_offset=phi_offset, phi_scale=phi_scale)

    def _cal_summand(_n: int, inv_fact: float) -> float:
        __dn_by_phi_func = dn_by_phi_func(n=_n, dn_a=cyl_dn_a,
                                          kb_t=kb_t, ks=ks,
                                          depth=depth, bias=bias,
                                          x_offset=x_offset, x_scale=x_scale,
                                          phi_offset=phi_offset, phi_scale=phi_scale)

        __pre = inv_fact * (1 / (_n + depth + 0.5))

        __der_x = scipy.misc.derivative(__dn_by_phi_func, x0=x, dx=1e-4)
        __der_x0 = scipy.misc.derivative(__dn_by_phi_func, x0=x0, dx=1e-4)
        __mittag = math.exp(-(_n + depth + 0.5) * (ks / friction_coeff) * t)

        return __pre * __der_x * __der_x0 * __mittag

    _sum: float = np.float128(0.0)

    # first term (n = 0)
    _sum += _cal_summand(0, 1)

    # Rest terms (n > 0)
    inv_factorial: float = 1.0
    for n in range(1, n_max):
        inv_factorial /= n
        _sum += _cal_summand(n, inv_factorial)

    _phi_x = _phi(x)
    _phi_x0 = _phi(x0)

    _first = c0_sq * math.sqrt(ks / kb_t) * (1 / (_phi_x ** 2))
    _sec = math.sqrt(kb_t / (2 * math.pi * ks)) * (_phi_x0 ** 2) * _sum

    return _first + _sec


# Vectorized Conditional Probability
def cond_prob_vec(x: np.ndarray | float, t: np.ndarray | float,
                  x0: np.ndarray | float, t0: np.ndarray | float,
                  n_max: np.ndarray | int,
                  cyl_dn_a: np.ndarray | float,
                  kb_t: np.ndarray | float,
                  ks: np.ndarray | float,
                  friction_coeff: np.ndarray | float,
                  depth: np.ndarray | float,
                  bias: np.ndarray | float,
                  x_offset: np.ndarray | float = 0,
                  x_scale: np.ndarray | float = 1,
                  phi_offset: np.ndarray | float = 0,
                  phi_scale: np.ndarray | float = 1) -> np.ndarray | np.float128:
    _vec = np.vectorize(cond_prob, otypes=[np.float128])
    return _vec(x=x, t=t, x0=x0, t0=t0,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


## Definte Integral of cond_prob over x
def cond_prob_integral_x(x0: float,
                         t0: float, t: float,
                         x_a: float, x_b: float, x_samples: int,
                         n_max: int,
                         cyl_dn_a: float,
                         kb_t: float,
                         ks: float,
                         friction_coeff: float,
                         depth: float,
                         bias: float,
                         x_offset: float = 0,
                         x_scale: float = 1,
                         phi_offset: float = 0,
                         phi_scale: float = 1) -> np.float128:
    x_arr = np.linspace(x_a, x_b, num=x_samples, endpoint=True)
    y_arr = cond_prob_vec(x=x_arr, t=t, x0=x0, t0=t0,
                          n_max=n_max, cyl_dn_a=cyl_dn_a,
                          kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                          depth=depth, bias=bias,
                          x_offset=x_offset, x_scale=x_scale,
                          phi_offset=phi_offset, phi_scale=phi_scale)

    return scipy.integrate.trapezoid(y_arr, x_arr)


def cond_prob_integral_x_vec(x0: np.ndarray | float,
                             t0: np.ndarray | float, t: np.ndarray | float,
                             x_a: np.ndarray | float, x_b: np.ndarray | float, x_samples: np.ndarray | int,
                             n_max: np.ndarray | int,
                             cyl_dn_a: np.ndarray | float,
                             kb_t: np.ndarray | float,
                             ks: np.ndarray | float,
                             friction_coeff: np.ndarray | float,
                             depth: np.ndarray | float,
                             bias: np.ndarray | float,
                             x_offset: np.ndarray | float = 0,
                             x_scale: np.ndarray | float = 1,
                             phi_offset: np.ndarray | float = 0,
                             phi_scale: np.ndarray | float = 1) -> np.ndarray | np.float128:
    _vec = np.vectorize(cond_prob_integral_x, otypes=[np.float128])
    return _vec(x0=x0,
                t0=t0, t=t,
                x_a=x_a, x_b=x_b, x_samples=x_samples,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def first_pass_time(x0: float,
                    t0: float, t: float,
                    x_a: float, x_b: float, x_samples: int,
                    n_max: int,
                    cyl_dn_a: float,
                    kb_t: float,
                    ks: float,
                    friction_coeff: float,
                    depth: float,
                    bias: float,
                    x_offset: float = 0,
                    x_scale: float = 1,
                    phi_offset: float = 0,
                    phi_scale: float = 1):
    def _f(time: float):
        return cond_prob_integral_x(x0=x0,
                                    t0=t0, t=time,
                                    x_a=x_a, x_b=x_b, x_samples=x_samples,
                                    n_max=n_max, cyl_dn_a=cyl_dn_a,
                                    kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                                    depth=depth, bias=bias,
                                    x_offset=x_offset, x_scale=x_scale,
                                    phi_offset=phi_offset, phi_scale=phi_scale)

    return -scipy.misc.derivative(_f, x0=t, dx=1e-10)  # TODO: derivative time step


def first_pass_time_vec(x0: np.ndarray | float,
                        t0: np.ndarray | float, t: np.ndarray | float,
                        x_a: np.ndarray | float, x_b: np.ndarray | float, x_samples: np.ndarray | int,
                        n_max: np.ndarray | int,
                        cyl_dn_a: np.ndarray | float,
                        kb_t: np.ndarray | float,
                        ks: np.ndarray | float,
                        friction_coeff: np.ndarray | float,
                        depth: np.ndarray | float,
                        bias: np.ndarray | float,
                        x_offset: np.ndarray | float = 0,
                        x_scale: np.ndarray | float = 1,
                        phi_offset: np.ndarray | float = 0,
                        phi_scale: np.ndarray | float = 1) -> np.ndarray | np.float128:
    _vec = np.vectorize(first_pass_time, otypes=[np.float128])
    return _vec(x0=x0,
                t0=t0, t=t,
                x_a=x_a, x_b=x_b, x_samples=x_samples,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def sp(x0: float, t0: float,
       t_start: float, t_stop: float, t_samples: int,
       x_a: float, x_b: float, x_samples: int,
       n_max: int,
       cyl_dn_a: float,
       kb_t: float,
       ks: float,
       friction_coeff: float,
       depth: float,
       bias: float,
       x_offset: float = 0,
       x_scale: float = 1,
       phi_offset: float = 0,
       phi_scale: float = 1) -> np.float128:
    t_arr = np.linspace(t_start, t_stop, num=t_samples, endpoint=True)
    fpt_arr = first_pass_time_vec(x0, t0=t0, t=t_arr, x_a=x_a, x_b=x_b, x_samples=x_samples,
                                  n_max=n_max, cyl_dn_a=cyl_dn_a,
                                  kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                                  depth=depth, bias=bias,
                                  x_offset=x_offset, x_scale=x_scale,
                                  phi_offset=phi_offset, phi_scale=phi_scale)

    return scipy.integrate.trapezoid(fpt_arr, t_arr)


def sp_vec(x0: np.ndarray | float,
           t0: np.ndarray | float,
           t_start: np.ndarray | float, t_stop: np.ndarray | float, t_samples: np.ndarray | int,
           x_a: np.ndarray | float, x_b: np.ndarray | float, x_samples: np.ndarray | int,
           n_max: np.ndarray | int,
           cyl_dn_a: np.ndarray | float,
           kb_t: np.ndarray | float,
           ks: np.ndarray | float,
           friction_coeff: np.ndarray | float,
           depth: np.ndarray | float,
           bias: np.ndarray | float,
           x_offset: np.ndarray | float = 0,
           x_scale: np.ndarray | float = 1,
           phi_offset: np.ndarray | float = 0,
           phi_scale: np.ndarray | float = 1) -> np.ndarray | np.float128:
    _vec = np.vectorize(sp, otypes=[np.float128])
    return _vec(x0=x0, t0=t0,
                t_start=t_start, t_stop=t_stop, t_samples=t_samples,
                x_a=x_a, x_b=x_b, x_samples=x_samples,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def test():
    pass


def test_mittag_leffler():
    x = np.linspace(0, 10, 200)
    y = mittag_leffler_vec(x, 1, 1, k_max=50)

    plt.plot(x, y, label="ML")
    plt.plot(x, np.exp(x), label="EXP")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    test()
