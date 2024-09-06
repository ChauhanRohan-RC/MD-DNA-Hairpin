import warnings

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import C
from C import COL_NAME_X, COL_NAME_SP_INTEGRAND, COL_NAME_SP, COL_NAME_PMF_RECONSTRUCTED, mp_execute, to_csv, \
    COL_NAME_PDF_RECONSTRUCTED
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


def mittag_leffler(x: int | float, a: int | float, b: int | float, k_max: int = 10) -> np.longdouble:
    # Special Cases: causes Overflow
    # if a == 1 and b == 1:
    #     return np.longdouble(math.exp(x))

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
    # Special Cases: causes Overflow
    # if a == 1 and b == 1:
    #     return np.exp(x, dtype=np.longdouble)

    _vec = np.vectorize(mittag_leffler, otypes=[np.longdouble])
    return _vec(x=x, a=a, b=b, k_max=k_max)


def critical_bias(depth: np.ndarray | float) -> np.ndarray | float:
    return math.sqrt(2) * (scipy.special.gamma((depth / 2) + 0.75) / scipy.special.gamma((depth / 2) + 0.25))


def dn(x: np.ndarray | float, n: int, a: float, hermite_monic=True,
       x_offset: float = 0, x_scale: float = 1,
       out_offset: float = 0, out_scale: float = 1
       ) -> np.ndarray | float:
    _hermite_poly = scipy.special.hermite(n, monic=hermite_monic)
    x = (x + x_offset) * x_scale
    _val = math.exp2(-n / 2) * np.exp(-a * np.square(x) / 2) * _hermite_poly(x * math.sqrt(a))
    return (_val + out_offset) * out_scale


def kn(n: np.ndarray | int, depth: np.ndarray | float,
       ks: np.ndarray | float, friction_coeff: np.ndarray | float) -> np.ndarray | float:
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


# Boltzmann Factor (Probability Density Function) from PMF (assuming Thermal Equilibrium)
def boltzmann_factor(pmf: np.ndarray | float, kb_t: float, normalize: bool = True, scale: float = 1) -> np.ndarray:
    pdf_arr = scale * np.exp(-pmf / kb_t)
    if normalize and isinstance(pmf, np.ndarray):
        s = np.sum(pdf_arr)
        if s != 0:
            pdf_arr /= s

    return pdf_arr


def pmf_from_pdf(pdf: np.ndarray, x: np.ndarray | None,
                 out_file_name: str | None,
                 kb_t: float,
                 out_x_col_name: str = COL_NAME_X,
                 out_pmf_col_name: str = C.COL_NAME_PMF) -> np.ndarray:
    """
    Boltzmann Inversion: PMF from equilibrium distribution
    """
    pmf_arr = -kb_t * np.log(pdf)

    if out_file_name:
        _df = pd.DataFrame()
        if x is not None:
            _df[out_x_col_name] = x
        _df[out_pmf_col_name] = pmf_arr
        to_csv(_df, out_file_name)

    return pmf_arr

def pdf_from_pmf(pmf: np.ndarray, x: np.ndarray | None,
                 out_file_name: str | None,
                 kb_t: float,
                 normalize: bool = True,
                 scale: float = 1,
                 out_x_col_name: str = COL_NAME_X,
                 out_pdf_col_name: str = COL_NAME_PDF_RECONSTRUCTED) -> np.ndarray | None:
    """
    Reconstructs Probability Density Function from PMF
    """
    pdf_arr = boltzmann_factor(pmf=pmf, kb_t=kb_t,
                               normalize=normalize, scale=scale)

    if out_file_name:
        _df = pd.DataFrame()
        if x is not None:
            _df[out_x_col_name] = x
        _df[out_pdf_col_name] = pdf_arr
        to_csv(_df, out_file_name)

    return pdf_arr


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

# Normalization Constant C0^2
def c0_sq(depth: float, bias: float) -> float:
    bias_crit = critical_bias(depth=depth)
    return ((bias_crit ** 2) - (bias ** 2)) / (2 * bias_crit)


# Normalization Constant Cn^2
def cn_sq(n: float, depth: float) -> float:
    a = math.sqrt(2 * math.pi) * C.factorial_cached(n - 1) * ((n - 1) + depth + 0.5)
    return 1 / a


def cond_prob(x: float, t: float, x0: float, t0: float,
              n_max: int,
              cyl_dn_a: float,
              kb_t: float,
              ks: float,
              beta: float,  # homogeneity coefficient beta [0, 1]
              friction_coeff_beta: float,
              depth: float,
              bias: float,
              x_offset: float = 0,
              x_scale: float = 1,
              phi_offset: float = 0,
              phi_scale: float = 1) -> np.longdouble:
    def _phi(_x):
        return phi_scaled(_x, kb_t=kb_t, ks=ks,
                          depth=depth, bias=bias,
                          x_offset=x_offset, x_scale=x_scale,
                          phi_offset=phi_offset, phi_scale=phi_scale)

    def _cal_summand(_n: int) -> float:
        __dn_by_phi_func = dn_by_phi_func(n=_n, dn_a=cyl_dn_a,
                                          kb_t=kb_t, ks=ks,
                                          depth=depth, bias=bias,
                                          x_offset=x_offset, x_scale=x_scale,
                                          phi_offset=phi_offset, phi_scale=phi_scale)

        # __pre = inv_fact * (1 / (_n + depth + 0.5))
        __pre = cn_sq(n=_n + 1, depth=depth)

        __der_x = derivative(__dn_by_phi_func, x0=x, dx=1e-4)
        __der_x0 = derivative(__dn_by_phi_func, x0=x0, dx=1e-4)

        if abs(beta - 1) <= 1e-4:  # tolerance
            __mittag = math.exp(-(_n + depth + 0.5) * (ks / friction_coeff_beta) * t)
        else:
            __mittag = mittag_leffler((ks / friction_coeff_beta) * (t ** beta), a=beta, b=1) ** -(_n + depth + 0.5)

        return __pre * __der_x * __der_x0 * __mittag

    _phi_x = _phi(x)
    _phi_x0 = _phi(x0)
    _ks_by_kbt_sqrt = math.sqrt(ks / kb_t)

    # First Term
    _first = c0_sq(depth=depth, bias=bias) * _ks_by_kbt_sqrt * (1 / (_phi_x ** 2))

    # Series
    _sum: float = np.longdouble(0.0)
    for n in range(0, n_max):
        _sum += _cal_summand(n)

    _sec = (1 / _ks_by_kbt_sqrt) * (_phi_x0 ** 2) * _sum

    return _first + _sec


# Vectorized Conditional Probability
def cond_prob_vec(x: np.ndarray | float, t: np.ndarray | float,
                  x0: np.ndarray | float, t0: np.ndarray | float,
                  normalize: bool,
                  n_max: np.ndarray | int,
                  cyl_dn_a: np.ndarray | float,
                  kb_t: np.ndarray | float,
                  ks: np.ndarray | float,
                  beta: np.ndarray | float,  # homogeneity coefficient beta [0, 1]
                  friction_coeff_beta: np.ndarray | float,
                  depth: np.ndarray | float,
                  bias: np.ndarray | float,
                  x_offset: np.ndarray | float = 0,
                  x_scale: np.ndarray | float = 1,
                  phi_offset: np.ndarray | float = 0,
                  phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(cond_prob, otypes=[np.longdouble])
    y_arr = _vec(x=x, t=t, x0=x0, t0=t0,
                 n_max=n_max, cyl_dn_a=cyl_dn_a,
                 kb_t=kb_t, ks=ks,
                 beta=beta, friction_coeff_beta=friction_coeff_beta,
                 depth=depth, bias=bias,
                 x_offset=x_offset, x_scale=x_scale,
                 phi_offset=phi_offset, phi_scale=phi_scale)

    if normalize and isinstance(x, np.ndarray):
        y_arr -= np.min(y_arr)
        tot = scipy.integrate.trapezoid(y=y_arr, x=x)
        if tot > 0:
            y_arr /= tot

        # y_arr -= np.min(y_arr)

    return y_arr


## Definite Integral of cond_prob over x
def cond_prob_integral_x(x0: float,
                         t0: float, t: float,
                         x_a: float, x_b: float, x_samples: int,
                         n_max: int,
                         cyl_dn_a: float,
                         kb_t: float,
                         ks: float,
                         beta: float,  # homogeneity coefficient beta [0, 1]
                         friction_coeff_beta: float,
                         depth: float,
                         bias: float,
                         x_offset: float = 0,
                         x_scale: float = 1,
                         phi_offset: float = 0,
                         phi_scale: float = 1) -> np.longdouble:
    x_arr = np.linspace(x_a, x_b, num=x_samples, endpoint=True)
    y_arr = cond_prob_vec(x=x_arr, t=t, x0=x0, t0=t0, normalize=False,
                          n_max=n_max, cyl_dn_a=cyl_dn_a,
                          kb_t=kb_t, ks=ks,
                          beta=beta, friction_coeff_beta=friction_coeff_beta,
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
                             beta: np.ndarray | float,  # homogeneity coefficient beta [0, 1]
                             friction_coeff_beta: np.ndarray | float,
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
                kb_t=kb_t, ks=ks,
                beta=beta, friction_coeff_beta=friction_coeff_beta,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def first_pass_time_first_princ(x0: float,
                                t0: float, t: float,
                                x_a: float, x_b: float, x_samples: int,
                                n_max: int,
                                cyl_dn_a: float,
                                kb_t: float,
                                ks: float,
                                beta: float,  # homogeneity coefficient beta [0, 1]
                                friction_coeff_beta: float,
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
                                    kb_t=kb_t, ks=ks,
                                    beta=beta, friction_coeff_beta=friction_coeff_beta,
                                    depth=depth, bias=bias,
                                    x_offset=x_offset, x_scale=x_scale,
                                    phi_offset=phi_offset, phi_scale=phi_scale)

    return -derivative(_f, x0=t, dx=1e-10)  # TODO: derivative time step


def first_pass_time_first_princ_vec(x0: np.ndarray | float,
                                    t0: np.ndarray | float, t: np.ndarray | float,
                                    x_a: np.ndarray | float, x_b: np.ndarray | float, x_samples: np.ndarray | int,
                                    n_max: np.ndarray | int,
                                    cyl_dn_a: np.ndarray | float,
                                    kb_t: np.ndarray | float,
                                    ks: np.ndarray | float,
                                    beta: np.ndarray | float,  # homogeneity coefficient beta [0, 1]
                                    friction_coeff_beta: np.ndarray | float,
                                    depth: np.ndarray | float,
                                    bias: np.ndarray | float,
                                    x_offset: np.ndarray | float = 0,
                                    x_scale: np.ndarray | float = 1,
                                    phi_offset: np.ndarray | float = 0,
                                    phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(first_pass_time_first_princ, otypes=[np.longdouble])
    return _vec(x0=x0,
                t0=t0, t=t,
                x_a=x_a, x_b=x_b, x_samples=x_samples,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks,
                beta=beta, friction_coeff_beta=friction_coeff_beta,
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
                               beta: float,  # homogeneity coefficient beta [0, 1]
                               friction_coeff_beta: float,
                               depth: float,
                               bias: float,
                               x_offset: float = 0,
                               x_scale: float = 1,
                               phi_offset: float = 0,
                               phi_scale: float = 1) -> np.longdouble:
    t_arr = np.linspace(t_start, t_stop, num=t_samples, endpoint=True)
    fpt_arr = first_pass_time_first_princ_vec(x0, t0=t0, t=t_arr, x_a=x_a, x_b=x_b, x_samples=x_samples,
                                              n_max=n_max, cyl_dn_a=cyl_dn_a,
                                              kb_t=kb_t, ks=ks,
                                              beta=beta, friction_coeff_beta=friction_coeff_beta,
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
                                   beta: np.ndarray | float,  # homogeneity coefficient beta [0, 1]
                                   friction_coeff_beta: np.ndarray | float,
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
                kb_t=kb_t, ks=ks,
                beta=beta, friction_coeff_beta=friction_coeff_beta,
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
                       beta: float,  # homogeneity coefficient beta [0, 1]
                       friction_coeff_beta: float,
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
                                    kb_t, ks,
                                    beta, friction_coeff_beta,
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

def first_pass_time_final_eq(x0: float, t: float,
                             x_a: float, x_b: float,
                             n_max: int,
                             cyl_dn_a: float,
                             kb_t: float,
                             ks: float,
                             beta: float,  # homogeneity coefficient beta -> [0, 1]
                             friction_coeff_beta: float,
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

    _d_beta = kb_t / friction_coeff_beta
    _phi_x0 = _phi(x0)

    _mittag_x = (ks / friction_coeff_beta) * (t ** beta)

    _pre: float = _d_beta * math.sqrt(ks / kb_t) * (_phi_x0 ** 2) * (t ** (beta - 1)) * mittag_leffler(x=_mittag_x,
                                                                                                       a=beta, b=beta)

    def _cal_summand(_n: int) -> float:
        __dn_by_phi_func = dn_by_phi_func(n=_n, dn_a=cyl_dn_a,
                                          kb_t=kb_t, ks=ks,
                                          depth=depth, bias=bias,
                                          x_offset=x_offset, x_scale=x_scale,
                                          phi_offset=phi_offset, phi_scale=phi_scale)

        __pre = cn_sq(n=_n + 1, depth=depth) * (n + depth + 0.5)

        __der = derivative(__dn_by_phi_func, x0=x0, dx=1e-4)
        __c = __dn_by_phi_func(x_b) - __dn_by_phi_func(x_a)

        if abs(beta - 1) <= 1e-4:  # tolerance
            __mittag = math.exp(-(_n + depth + 1.5) * _mittag_x)
        else:
            __mittag = mittag_leffler(_mittag_x, a=beta, b=1) ** -(_n + depth + 1.5)

        return __pre * __der * __c * __mittag

    _sum = np.longdouble(0)
    for n in range(0, n_max):
        _sum += _cal_summand(n)

    return _pre * _sum


def first_pass_time_final_eq_vec(x0: np.ndarray | float, t: np.ndarray | float,
                                 x_a: np.ndarray | float, x_b: np.ndarray | float,
                                 n_max: np.ndarray | int,
                                 cyl_dn_a: np.ndarray | float,
                                 kb_t: np.ndarray | float,
                                 ks: np.ndarray | float,
                                 beta: np.ndarray | float,  # homogeneity coefficient beta -> [0, 1]
                                 friction_coeff_beta: np.ndarray | float,
                                 depth: np.ndarray | float,
                                 bias: np.ndarray | float,
                                 x_offset: np.ndarray | float = 0,
                                 x_scale: np.ndarray | float = 1,
                                 phi_offset: np.ndarray | float = 0,
                                 phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(first_pass_time_final_eq, otypes=[np.longdouble])

    return _vec(x0=x0, t=t,
                x_a=x_a, x_b=x_b,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks,
                beta=beta, friction_coeff_beta=friction_coeff_beta,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


def mean_first_pass_time_final_eq(x0: float,
                                  x_a: float, x_b: float,
                                  n_max: int,
                                  cyl_dn_a: float,
                                  kb_t: float,
                                  ks: float,
                                  beta: float,  # homogeneity coefficient beta -> [0, 1]
                                  friction_coeff_beta: float,
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

    _phi_x0 = _phi(x0)
    _pre = math.sqrt(kb_t / ks) * (_phi_x0 ** 2) * ((ks / friction_coeff_beta) ** beta) * scipy.special.gamma(1 + beta)

    def _cal_summand(_n: int) -> float:
        __dn_by_phi_func = dn_by_phi_func(n=_n, dn_a=cyl_dn_a,
                                          kb_t=kb_t, ks=ks,
                                          depth=depth, bias=bias,
                                          x_offset=x_offset, x_scale=x_scale,
                                          phi_offset=phi_offset, phi_scale=phi_scale)

        __pre = cn_sq(n=_n + 1, depth=depth) / (n + depth + 0.5)
        __der = derivative(__dn_by_phi_func, x0=x0, dx=1e-4)
        __c = __dn_by_phi_func(x_b) - __dn_by_phi_func(x_a)
        return __pre * __der * __c

    _sum = np.longdouble(0)
    for n in range(0, n_max):
        _sum += _cal_summand(n)

    return _pre * _sum


def mean_first_pass_time_final_eq_vec(x0: np.ndarray | float,
                                      x_a: np.ndarray | float, x_b: np.ndarray | float,
                                      n_max: np.ndarray | int,
                                      cyl_dn_a: np.ndarray | float,
                                      kb_t: np.ndarray | float,
                                      ks: np.ndarray | float,
                                      beta: np.ndarray | float,  # homogeneity coefficient beta -> [0, 1]
                                      friction_coeff_beta: np.ndarray | float,
                                      depth: np.ndarray | float,
                                      bias: np.ndarray | float,
                                      x_offset: np.ndarray | float = 0,
                                      x_scale: np.ndarray | float = 1,
                                      phi_offset: np.ndarray | float = 0,
                                      phi_scale: np.ndarray | float = 1) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(mean_first_pass_time_final_eq, otypes=[np.longdouble])

    return _vec(x0=x0,
                x_a=x_a, x_b=x_b,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks,
                beta=beta, friction_coeff_beta=friction_coeff_beta,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale)


# The integrand of Splitting probability from final equation
# This must be integrated over x in a running-manner to get final Splitting Probability
def __sp_final_eq_integrand(x0: float,
                            x_a: float, x_b: float,
                            n_max: int,
                            cyl_dn_a: float,
                            kb_t: float,
                            ks: float,
                            beta: float,  # homogeneity coefficient beta -> [0, 1]
                            friction_coeff_beta: float,
                            depth: float,
                            bias: float,
                            x_offset: float = 0,
                            x_scale: float = 1,
                            phi_offset: float = 0,
                            phi_scale: float = 1,
                            t_integration_start: float = 0,  # only when beta != 1
                            t_integration_stop: float = 1e-4,  # only when beta != 1
                            t_integration_samples: int = 100  # only when beta != 1
                            ) -> np.longdouble:
    def _phi(_x: np.ndarray | float):
        return phi_scaled(_x, kb_t=kb_t, ks=ks,
                          depth=depth, bias=bias,
                          x_offset=x_offset, x_scale=x_scale,
                          phi_offset=phi_offset, phi_scale=phi_scale)

    _d_beta = kb_t / friction_coeff_beta
    _phi_x0 = _phi(x0)

    _pre: float = _d_beta * math.sqrt(ks / (2 * math.pi * kb_t)) * (_phi_x0 ** 2)

    def _cal_summand(_n: int) -> float:
        __dn_by_phi_func = dn_by_phi_func(n=_n, dn_a=cyl_dn_a,
                                          kb_t=kb_t, ks=ks,
                                          depth=depth, bias=bias,
                                          x_offset=x_offset, x_scale=x_scale,
                                          phi_offset=phi_offset, phi_scale=phi_scale)

        __pre = 1 / C.factorial_cached(_n)

        __der = derivative(__dn_by_phi_func, x0=x0, dx=1e-4)
        __c = __dn_by_phi_func(x_b) - __dn_by_phi_func(x_a)
        if abs(beta - 1) <= 1e-4:  # tolerance
            __integral = 1 / kn(n=_n, depth=depth, ks=ks, friction_coeff=friction_coeff_beta)
        else:
            _t = np.linspace(t_integration_start, t_integration_stop, num=t_integration_samples, endpoint=True)
            _x = (ks / friction_coeff_beta) * np.power(_t, beta)
            __y = mittag_leffler_vec(x=_x, a=beta, b=1)
            _y = np.power(__y, -(_n + depth + 0.5))

            __integral = scipy.integrate.trapezoid(y=_y, x=_t)

        return __der * __c * __integral

    _sum = np.longdouble(0)

    for n in range(0, n_max):
        _sum += _cal_summand(n)

    return _pre * _sum


def _sp_final_eq_integrand_vec(x0: np.ndarray | float,
                               x_a: np.ndarray | float, x_b: np.ndarray | float,
                               n_max: np.ndarray | int,
                               cyl_dn_a: np.ndarray | float,
                               kb_t: np.ndarray | float,
                               ks: np.ndarray | float,
                               beta: np.ndarray | float,  # homogeneity coefficient beta [0, 1]
                               friction_coeff_beta: np.ndarray | float,
                               depth: np.ndarray | float,
                               bias: np.ndarray | float,
                               x_offset: np.ndarray | float = 0,
                               x_scale: np.ndarray | float = 1,
                               phi_offset: np.ndarray | float = 0,
                               phi_scale: np.ndarray | float = 1,
                               t_integration_start: float = 0,  # only when beta != 1
                               t_integration_stop: float = 1e-4,  # only when beta != 1
                               t_integration_samples: int = 100  # only when beta != 1
                               ) -> np.ndarray | np.longdouble:
    _vec = np.vectorize(__sp_final_eq_integrand, otypes=[np.longdouble])
    return _vec(x0=x0,
                x_a=x_a, x_b=x_b,
                n_max=n_max, cyl_dn_a=cyl_dn_a,
                kb_t=kb_t, ks=ks,
                beta=beta, friction_coeff_beta=friction_coeff_beta,
                depth=depth, bias=bias,
                x_offset=x_offset, x_scale=x_scale,
                phi_offset=phi_offset, phi_scale=phi_scale,
                t_integration_start=t_integration_start,
                t_integration_stop=t_integration_stop,
                t_integration_samples=t_integration_samples)


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
                beta: np.ndarray | float,  # homogeneity coefficient beta [0, 1]
                friction_coeff_beta: float,
                depth: float,
                bias: float,
                x_offset: float = 0,
                x_scale: float = 1,
                phi_offset: float = 0,
                phi_scale: float = 1,
                t_integration_start: float = 0,  # only when beta != 1
                t_integration_stop: float = 1e-4,  # only when beta != 1
                t_integration_samples: int = 100  # only when beta != 1
                ) -> pd.DataFrame:
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
                                    kb_t, ks,
                                    beta, friction_coeff_beta,
                                    depth, bias,
                                    x_offset, x_scale,
                                    phi_offset, phi_scale,
                                    t_integration_start, t_integration_stop, t_integration_samples))

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
    pmf_arr = double_well_pmf_scaled(x=x,
                                     kb_t=kb_t, ks=ks,
                                     depth=depth, bias=bias,
                                     x_offset=x_offset, x_scale=x_scale,
                                     phi_offset=phi_offset, phi_scale=phi_scale)
    return np.exp(pmf_arr / kb_t)


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
                phi_scale: float = 1) -> pd.DataFrame:
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


def sp_apparent2(x: np.ndarray, pmf: np.ndarray,
                return_sp_integrand: bool,
                reconstruct_pmf: bool,
                out_data_file: str | None,
                kb_t: float) -> pd.DataFrame:
    """
    Splitting Probability directly from Apparent PMF samples.
    """

    sp_integrand = np.exp(pmf / kb_t)

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
    test_mittag_leffler()
