import math
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from C import COL_NAME_X, COL_NAME_SP_INTEGRAND, COL_NAME_SP, COL_NAME_PMF_RECONSTRUCTED, mp_execute, to_csv
from double_well_pmf import phi_scaled, double_well_pmf_scaled

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
FIRST-PRINCIPLE and FINAL-EQUATION implementations of Splitting Probability from the ground up

## FIRST PRINCIPLE Implementation ----------------------
1. cond_prob(x, t, x0, t0)
    conditional probability  P(x,t | x0,t0) from generalized fokker-planck equation
    
2. cond_prob_int_x(t, x0, t0)
    conditional probability integral w.r.t to x at given t, x0, t0

3. fpt(x0, t)
    First Passage Time Distribution FPT(x0, t) = -ve time-gradient of cond_prob_int_x(t, x0, t0)

4. sp_first_principle(x0)
    Splitting probability as running integral of [0th time-moment of FPT]
    -> Sp(x0) = running_integral( time_integral(FPT(x0, t)) )

## FINAL-EQUATION "EXACT" Implementation ----------------------
1. sp_final_eq(x0)
    Splitting probability from final Equation [NO INTEGRALS OR CALCULUS]
    Much more accurate and faster than first principle implementation
    
## APPARENT-EQUILIBRIUM Implementation ----------------------
1. sp_app(x)
    Splitting Probability from Apparent PMF implied by extension distribution
    (assumes equilibrium) using Boltzmann inversion

NOTE: functions ending with "vec" are vectorized versions of corresponding functions.
      They accept array of inputs, and compute the base function at each input in a for-loop
"""

# CONSTANTS -------------------------------------------------------------
OFFSET_SP_INTEGRAND_TO_POSITIVE: bool = True
TRANSFORM_Dn_OUT_AS_PHI: bool = True

"""
FLags controlling the behaviour of final step in SP calculation
    
    if true, 
        final entity in calculation will be considered as "SP_INTEGRAND" which
        will be integrated over x in a running fashion to give SP
    
    Otherwise:
        final entity in calculation will be considered SP as it is.
        NO running-integral over x will be performed
"""
RUNNING_INTEGRAL_SP_FIRST_PRINCIPLE: bool = True
RUNNING_INTEGRAL_SP_FINAL_EQ: bool = True
RUNNING_INTEGRAL_SP_APP_PMF: bool = True


# ==========================================================================
# ------------------- UTILITY FUNCTIONS  -----------------------
# ==========================================================================

def derivative(func, x0: float, dx: float):
    """
    :param func: the function to be differentiated
    :param x0: point at which derivative is computed
    :param dx: derivative step size. Must be as low as possible
    :return: the derivative of the function at x0
    """
    return scipy.misc.derivative(func, x0=x0, dx=dx)


def mittag_leffler(x: int | float, a: int | float, b: int | float, k_max: int = 50) -> np.longdouble:
    _sum = np.longdouble(0)

    _x_pow = np.longdouble(1)
    for k in range(0, k_max):
        _val = _x_pow / scipy.special.gamma((a * k) + b)

        _sum += _val
        _x_pow *= x

    return _sum


def mittag_leffler_vec(x: np.ndarray | int | float,
                       a: np.ndarray | int | float,
                       b: np.ndarray | int | float,
                       k_max: np.ndarray | int = 50) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(mittag_leffler, otypes=[np.longdouble])
    return _vec(x=x, a=a, b=b, k_max=k_max)


def dn(x: np.ndarray | float, n: int, a: float, hermite_monic=True,
       x_offset: float = 0, x_scale: float = 1,
       out_offset: float = 0, out_scale: float = 1
       ) -> np.ndarray | float:
    _hermite_poly = scipy.special.hermite(n, monic=hermite_monic)
    x = (x + x_offset) * x_scale
    _val = math.exp2(-n / 2) * np.exp(-a * np.square(x) / 2) * _hermite_poly(x * math.sqrt(a))
    return (_val + out_offset) * out_scale


def kn(n: int, depth: float, ks: float, friction_coeff: float) -> float:
    return (n + depth + 0.5) * ks / friction_coeff


def dn_by_phi(x: np.ndarray | float, n: int, dn_a: float,
              kb_t: float,
              ks: float,
              depth: float,
              bias: float,
              x_offset: float = 0,
              x_scale: float = 1,
              phi_offset: float = 0,
              phi_scale: float = 1) -> np.ndarray | float:

    dn_val = dn(x, n=n, a=dn_a,
                x_offset=x_offset, x_scale=x_scale,
                out_offset=phi_offset if TRANSFORM_Dn_OUT_AS_PHI else 0,
                out_scale=phi_scale if TRANSFORM_Dn_OUT_AS_PHI else 1)

    phi_val = phi_scaled(x, kb_t=kb_t, ks=ks,
                         depth=depth, bias=bias,
                         x_offset=x_offset, x_scale=x_scale,
                         phi_offset=phi_offset, phi_scale=phi_scale)

    return dn_val / phi_val


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


# ==========================================================================
# ------------------- COMMON METHODS  -----------------------
# ==========================================================================

# Reconstructs RMF from Splitting Probability
def pmf_re(x: np.ndarray, sp: np.ndarray, kb_t: float):
    _grad: np.ndarray = np.gradient(sp, x)
    print("SP IMPL: Reconstructing PMF from SP")
    print(f"SP IMPL: SP gradient +ve count: {(_grad > 0).sum()}")

    return kb_t * np.log(-_grad)


def _create_sp_dataframe(x: np.ndarray,
                         sp: np.ndarray,
                         sp_integrand: np.ndarray | None,
                         pmf_re: np.ndarray | None,
                         out_data_file: str | None):
    _df = pd.DataFrame()
    _df[COL_NAME_X] = x
    if sp_integrand is not None:
        _df[COL_NAME_SP_INTEGRAND] = sp_integrand
    _df[COL_NAME_SP] = sp

    if pmf_re is not None:
        _df[COL_NAME_PMF_RECONSTRUCTED] = pmf_re

    if out_data_file:
        print(f"SP IMPL: Writing Splitting Probability DataFrame to file \"{out_data_file}\"")
        to_csv(_df, out_data_file)

    return _df


def _handle_sp_integrand(x: np.ndarray,
                         sp_integrand: np.ndarray,
                         kb_t: float,
                         do_integrate: bool,
                         # whether to integrate the sp_integrand to calculate SP or just consider it SP as it is
                         return_sp_integrand: bool,
                         reconstruct_pmf: bool,
                         out_data_file: str | None) -> pd.DataFrame:
    _min = np.min(sp_integrand)
    # Offset to get only values
    if _min < 0:
        print(f"SP_IMPL: SP_INTEGRAND has negative values. Min Value: {_min}")
        if OFFSET_SP_INTEGRAND_TO_POSITIVE:
            sp_integrand -= _min
            print(f"SP_IMPL: Offsetting SP_INTEGRAND (to make it Positive) by the Min Value: {_min}")
        else:
            print(f"SP_IMPL: SP_INTEGRAND Offsetting disabled!! Working with negative values...")

    if do_integrate:
        # Integral in the denominator = Constant
        c = scipy.integrate.trapezoid(y=sp_integrand, x=x)
        sp = np.zeros(len(x), dtype=np.longdouble)

        for i in range(len(x)):
            _v = scipy.integrate.trapezoid(y=sp_integrand[i:], x=x[i:])
            sp[i] = _v / c
    else:
        sp = sp_integrand

    _sp_integrand = sp_integrand if do_integrate and return_sp_integrand else None
    _pmf_re = pmf_re(x=x, sp=sp, kb_t=kb_t) if reconstruct_pmf else None

    return _create_sp_dataframe(x=x,
                                sp=sp,
                                sp_integrand=_sp_integrand,
                                pmf_re=_pmf_re,
                                out_data_file=out_data_file)


# ==========================================================================
# ------------------- FIRST PRINCIPLE IMPLEMENTATION  -----------------------
# ==========================================================================

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
              phi_scale: float = 1) -> np.longdouble:
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

        __der_x = derivative(__dn_by_phi_func, x0=x, dx=1e-4)
        __der_x0 = derivative(__dn_by_phi_func, x0=x0, dx=1e-4)
        __mittag = math.exp(-(_n + depth + 0.5) * (ks / friction_coeff) * t)

        return __pre * __der_x * __der_x0 * __mittag

    _sum: float = np.longdouble(0.0)

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
                  phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(cond_prob, otypes=[np.longdouble])
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
                         phi_scale: float = 1) -> np.longdouble:
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
                             phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(cond_prob_integral_x, otypes=[np.longdouble])
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

    return -derivative(_f, x0=t, dx=1e-10)  # TODO: derivative time step


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
                        phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(first_pass_time, otypes=[np.longdouble])
    return _vec(x0=x0,
                t0=t0, t=t,
                x_a=x_a, x_b=x_b, x_samples=x_samples,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


# The integrand of Splitting probability = 0th moment of first_passage_time
# This must be integrated over x in a running-manner to get final Splitting Probability
# (uses defining integral equations)
def __sp_first_princ_integrand(x0: float, t0: float,
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
                               phi_scale: float = 1) -> np.longdouble:
    t_arr = np.linspace(t_start, t_stop, num=t_samples, endpoint=True)
    fpt_arr = first_pass_time_vec(x0, t0=t0, t=t_arr, x_a=x_a, x_b=x_b, x_samples=x_samples,
                                  n_max=n_max, cyl_dn_a=cyl_dn_a,
                                  kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                                  depth=depth, bias=bias,
                                  x_offset=x_offset, x_scale=x_scale,
                                  phi_offset=phi_offset, phi_scale=phi_scale)

    return scipy.integrate.trapezoid(fpt_arr, t_arr)


def __sp_first_princ_integrand_vec(x0: np.ndarray | float,
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
                                   phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(__sp_first_princ_integrand, otypes=[np.longdouble])
    return _vec(x0=x0, t0=t0,
                t_start=t_start, t_stop=t_stop, t_samples=t_samples,
                x_a=x_a, x_b=x_b, x_samples=x_samples,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def sp_first_principle(x_a: float, x_b: float,
                       x_integration_samples: int,
                       t0: float, t_start: float, t_stop: float, t_samples: int,
                       process_count: int,
                       return_sp_integrand: bool,
                       reconstruct_pmf: bool,
                       out_data_file: str | None,
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
                       phi_scale: float = 1) -> pd.DataFrame:
    """
    Calculates the Splitting probability between x_a and x_b.
    It's a "first-principle" implementation
    Uses the 0th moment of first passage time definition.

    HIGHLY COMPUTATION INTENSIVE: uses multiple sampling, differentiation and integration steps
    Use "sp_final_eq" for most purposes

    :returns A pandas dataframe containing following columns
        1. COL_NAME_X:
            -> X values sampled in range [x_a, x_b]. sample count = x_integration_samples
        2. COL_NAME_SP_INTEGRAND:
            -> Splitting Probability integrand values at x
            -> ONLY PRESENT WHEN "return_sp_integrand=True"
        3. COL_NAME_SP:
            -> splitting probability values at x
        4. COL_NAME_PMF_RE:
            -> PMF reconstructed from splitting probability. see {@link pme_re(x, sp_integrand, kb_t)}
            -> ONLY PRESENT WHEN "reconstruct_pmf=True"

        if "out_data_file" is set, the returned DataFrame is also saved to this file
    """
    x = np.linspace(x_a, x_b, x_integration_samples, endpoint=True)
    sp_integrand = mp_execute(__sp_first_princ_integrand_vec, input_arr=x, process_count=process_count,
                              args=(t0, t_start, t_stop, t_samples,
                                    x_a, x_b, x_integration_samples,
                                    n_max, cyl_dn_a,
                                    kb_t, ks, friction_coeff,
                                    depth, bias,
                                    x_offset, x_scale,
                                    phi_offset, phi_scale))

    return _handle_sp_integrand(x=x,
                                sp_integrand=sp_integrand,
                                kb_t=kb_t,
                                do_integrate=RUNNING_INTEGRAL_SP_FIRST_PRINCIPLE,
                                return_sp_integrand=return_sp_integrand,
                                reconstruct_pmf=reconstruct_pmf,
                                out_data_file=out_data_file)


# ==========================================================================
# ------------------- FINAL EQUATION IMPLEMENTATION  -----------------------
# ==========================================================================

# The integrand of Splitting probability from final equation
# This must be integrated over x in a running-manner to get final Splitting Probability
def __sp_final_eq_integrand(x0: float,
                            x_a: float, x_b: float,
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
                            phi_scale: float = 1) -> np.longdouble:
    def _phi(_x: np.ndarray | float):
        return phi_scaled(_x, kb_t=kb_t, ks=ks,
                          depth=depth, bias=bias,
                          x_offset=x_offset, x_scale=x_scale,
                          phi_offset=phi_offset, phi_scale=phi_scale)

    _d1 = kb_t / friction_coeff
    _pre_c: float = _d1 * math.sqrt(ks / (kb_t * 2 * math.pi))
    _phi_x = _phi(x0)

    _pre = _pre_c * _phi_x * _phi_x

    def _cal_summand(_n: int, inv_fact: float) -> float:
        __dn_by_phi_func = dn_by_phi_func(n=_n, dn_a=cyl_dn_a,
                                          kb_t=kb_t, ks=ks,
                                          depth=depth, bias=bias,
                                          x_offset=x_offset, x_scale=x_scale,
                                          phi_offset=phi_offset, phi_scale=phi_scale)

        __der = derivative(__dn_by_phi_func, x0=x0, dx=1e-4)
        __c = __dn_by_phi_func(x_b) - __dn_by_phi_func(x_a)
        __integral = 1 / kn(n=_n, depth=depth, ks=ks, friction_coeff=friction_coeff)

        return inv_fact * __der * __c * __integral

    _sum = np.longdouble(0)

    # first term (n = 0)
    _sum += _cal_summand(0, 1)

    # Rest terms (n > 1)
    inv_factorial: float = 1.0
    for n in range(1, n_max):
        inv_factorial /= n
        _sum += _cal_summand(n, inv_factorial)

    return _pre * _sum


def _sp_final_eq_integrand_vec(x0: np.ndarray | float,
                               x_a: np.ndarray | float, x_b: np.ndarray | float,
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
                               phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(__sp_final_eq_integrand, otypes=[np.longdouble])
    return _vec(x0=x0,
                x_a=x_a, x_b=x_b,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks, friction_coeff=friction_coeff,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def sp_final_eq(x_a: float, x_b: float,
                x_integration_samples: int,
                process_count: int,
                return_sp_integrand: bool,
                reconstruct_pmf: bool,
                out_data_file: str | None,
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
                phi_scale: float = 1) -> pd.DataFrame:
    """
    Calculates the Splitting probability between x_a and x_b.
    Uses the final "exact" equation, hence cheap and accurate.

    Use this method instead of first-principle implementation "sp"

    :returns A pandas dataframe containing following columns
        1. COL_NAME_X:
            -> X values sampled in range [x_a, x_b]. sample count = x_integration_samples
        2. COL_NAME_SP_INTEGRAND:
            -> Splitting Probability integrand values at x
            -> ONLY PRESENT WHEN "return_sp_integrand=True"
        3. COL_NAME_SP:
            -> splitting probability values at x
        4. COL_NAME_PMF_RE:
            -> PMF reconstructed from splitting probability. see {@link pme_re(x, sp_integrand, kb_t)}
            -> ONLY PRESENT WHEN "reconstruct_pmf=True"

        if "out_data_file" is set, the returned DataFrame is also saved to this file
    """
    x = np.linspace(x_a, x_b, num=x_integration_samples, endpoint=True)
    sp_integrand = mp_execute(_sp_final_eq_integrand_vec, input_arr=x, process_count=process_count,
                              args=(x_a, x_b,
                                    n_max, cyl_dn_a,
                                    kb_t, ks, friction_coeff,
                                    depth, bias,
                                    x_offset, x_scale,
                                    phi_offset, phi_scale))

    return _handle_sp_integrand(x=x,
                                sp_integrand=sp_integrand,
                                kb_t=kb_t,
                                do_integrate=RUNNING_INTEGRAL_SP_FINAL_EQ,
                                return_sp_integrand=return_sp_integrand,
                                reconstruct_pmf=reconstruct_pmf,
                                out_data_file=out_data_file)


# ==========================================================================
# -------------------------- APPARENT SP -----------------------------------
# ==========================================================================

def _sp_app_integrand_vec(x: np.ndarray | float,
                          kb_t: float,
                          ks: float,
                          depth: float,
                          bias: float,
                          x_offset: float = 0,
                          x_scale: float = 1,
                          phi_offset: float = 0,
                          phi_scale: float = 1):
    beta = 1 / kb_t
    pmf_arr = double_well_pmf_scaled(x=x,
                                     kb_t=kb_t, ks=ks,
                                     depth=depth, bias=bias,
                                     x_offset=x_offset, x_scale=x_scale,
                                     phi_offset=phi_offset, phi_scale=phi_scale)
    return np.exp(beta * pmf_arr)


def sp_apparent(x_a: float, x_b: float,
                x_integration_samples: int,
                process_count: int,
                return_sp_integrand: bool,
                reconstruct_pmf: bool,
                out_data_file: str | None,
                kb_t: float,
                ks: float,
                depth: float,
                bias: float,
                x_offset: float = 0,
                x_scale: float = 1,
                phi_offset: float = 0,
                phi_scale: float = 1):
    """
    Calculates the Splitting probability between x_a and x_b using Apparent-PMF implied by
    extension distribution.
    Assumes Equilibrium case, and uses Boltzmann inversion to calculate Sp(x) using the apparent pmf

        Apparent PMF(x) = - KbT * ln(Peq(x))   where Peq(x) is the equilibrium extension distribution

    :returns A pandas dataframe containing following columns
        1. COL_NAME_X:
            -> X values sampled in range [x_a, x_b]. sample count = x_integration_samples
        2. COL_NAME_SP_INTEGRAND:
            -> Splitting Probability integrand values at x
            -> ONLY PRESENT WHEN "return_sp_integrand=True"
        3. COL_NAME_SP:
            -> splitting probability values at x
        4. COL_NAME_PMF_RE:
            -> PMF reconstructed from splitting probability. see {@link pme_re(x, sp_integrand, kb_t)}
            -> ONLY PRESENT WHEN "reconstruct_pmf=True"

        if "out_data_file" is set, the returned DataFrame is also saved to this file
    """

    x = np.linspace(x_a, x_b, x_integration_samples, endpoint=True)
    sp_integrand = mp_execute(_sp_app_integrand_vec, input_arr=x, process_count=process_count,
                              args=(kb_t, ks,
                                    depth, bias,
                                    x_offset, x_scale,
                                    phi_offset, phi_scale))

    return _handle_sp_integrand(x=x,
                                sp_integrand=sp_integrand,
                                kb_t=kb_t,
                                do_integrate=RUNNING_INTEGRAL_SP_APP_PMF,
                                return_sp_integrand=return_sp_integrand,
                                reconstruct_pmf=reconstruct_pmf,
                                out_data_file=out_data_file)


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
