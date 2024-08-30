import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import C
from C import *

"""
Models Double-Well PMF with depth and bias parameters

THe Double-Well PMF is modelled using the quantity phi(A, B, x) where A is the depth parameter
and B is the bias parameter. It is given by the linear combination of even and odd solution of Weber equation

        Φ(A,x) = y1(A,x) + bias * y2(A,x)

The corresponding Probability Distribution and PMF are given as

        Probability distribution Peq(x) = phi(A,x)^-2
        PMF = -Kb * T * ln(Peq(x)) = 2 * Kb * T * ln(phi(A,x))
"""


def parabolic_pmf(x: np.ndarray,
                  ks: float,
                  x_offset: float = 0,
                  x_scale: float = 1,
                  pmf_offset: float = 0,
                  pmf_scale: float = 1,
                  out_file_name: str | None = None,
                  out_col_name_x: str = COL_NAME_X,
                  out_col_name_pmf: str = COL_NAME_PMF) -> np.ndarray:
    """
        Transform x ->  x = (x + x_offset) * x_scale

        Calculate PMF -> pmf = 0.5 * ks * x^2

        Transform pmf -> pmf = (pmf + pmf_Offset) * pmf_scale

    :return: the parabolic pmf, after offsetting and scaling
    """
    pmf = pmf_scale * (0.5 * ks * np.square((x + x_offset) * x_scale) + pmf_offset)

    if out_file_name:
        df = pd.DataFrame({
            out_col_name_x: x,
            out_col_name_pmf: pmf
        })

        with open(out_file_name, "w") as f:
            f.write(f"{COMMENT_TOKEN} ------------- PARABOLIC PMF ----------------\n")
            f.write(f"{COMMENT_TOKEN} INPUT Spring constant (Ks): {ks}\n")
            f.write(
                f"{COMMENT_TOKEN} Parameters => x_offset: {x_offset} | x_scale: {x_scale} | pmf_offset: {pmf_offset} | pmf_scale: {pmf_scale}\n")
            to_csv(df, path_or_buf=f, mode="a")

    return pmf


def phi(x: np.ndarray,
        kb_t: float,
        ks: float,
        depth: float,
        bias: float) -> np.ndarray:
    """
    Calculates Φ(A,x) that represents Double-Well Potential.
    It is given by the linear combination of even and odd solution of Weber equation

        Φ(A,x) = y1(A,x) + bias * y2(A,x)

    The corresponding Probability Distribution and PMF are given as

        Probability distribution Peq(x) = phi(A,x)^-2
        PMF = -Kb * T * ln(Peq(x)) = 2 * Kb * T * ln(phi(A,x))

    NOTE: x, KbT, Ks should be of matching units. Usually
    -> Å, kcal/mol, Kcal/mol/Å**2
    -> nm, pN nm, pN/nm

    :param x: the reaction coordinate as a numpy array
    :param depth: depth parameter of potential well, symbol A (dimensionless). generally in range [-0.5, 0)
    :param bias: bias parameter of double-well potential, symbol B (dimensionless)
    :param kb_t: thermal energy
    :param ks: spring constant representing curvature of energy barrier (or rigidity of optical trap)

    :return: Φ(A,x), as numpy array.
    """

    # Common terms
    _xsq: np.ndarray = np.square(x)
    _common: np.ndarray = np.exp(-(ks / (4 * kb_t)) * _xsq)

    _hyper1f1_coeff = ks / (2 * kb_t)
    _x_hyper = _hyper1f1_coeff * _xsq

    # Odd solution of weber equation
    if abs(depth + 0.5) <= 1e-10:  # Tolerance for depth
        _y1 = 1
    else:
        _y1 = scipy.special.hyp1f1((depth / 2) + 0.25, 0.5, _x_hyper)

    # Even solution of weber equation
    if abs(bias - 0) <= 1e-10:  # Tolerance for bias
        _y2 = 0
    else:
        _y2 = bias * math.sqrt(ks / kb_t) * x * scipy.special.hyp1f1((depth / 2) + 0.75, 1.5, _x_hyper)

    # Φ(A,x) = y1(A,x) + bias * y2(A,x)
    return _common * (_y1 + _y2)

    ## NOTE: Final Probability distribution Peq(x) = Φ(A,x)^-2
    # peq = 1 / np.square(_phi)


def double_well_pmf(x: np.ndarray,
                    kb_t: float,
                    ks: float,
                    depth: float,
                    bias: float) -> np.ndarray:
    """
    Helper method for calculating Potential of Mean Force (PMF) from Φ(A,x)

        PMF(x) = -KbT * ln(Peq(x)) = 2 * KbT * ln(Φ(A,x))

    {@inheritdoc phi}
    :return: Potential of Mean Force (PMF)
    """

    _phi_ax = phi(x=x,
                  kb_t=kb_t,
                  ks=ks,
                  depth=depth,
                  bias=bias)

    return 2 * kb_t * np.log(_phi_ax)


def phi_scaled(x: np.ndarray,
               kb_t: float,
               ks: float,
               depth: float,
               bias: float,
               x_offset: float = 0,
               x_scale: float = 1,
               phi_offset: float = 0,
               phi_scale: float = 1) -> np.ndarray:
    """
    Wrapper over phi(depth, bias, kb_t, ks) to support input and output transforms

    NOTE: Scaling

        x = (x + x_offset) * x_scale
        output = (Φ(A,x) + phi_offset) * phi_scale

    :param x_scale: scalar which is multiplied to the given x
    :param x_offset: scalar which is added to the scaled x
    :param phi_scale: scalar which is multiplied to the output Φ(A,x)
    :param phi_offset: scalar which is added to the scaled Φ(A,x)

    :return: Φ(A,x) after scaling, as numpy array.
    """

    # Scaling Input
    x = (x + x_offset) * x_scale

    _phi = phi(x=x,
               kb_t=kb_t,
               ks=ks,
               depth=depth,
               bias=bias)

    # return scaled output
    return (_phi + phi_offset) * phi_scale


def double_well_pmf_scaled(x: np.ndarray,
                           kb_t: float,
                           ks: float,
                           depth: float,
                           bias: float,
                           x_offset: float = 0,
                           x_scale: float = 1,
                           phi_offset: float = 0,
                           phi_scale: float = 1) -> np.ndarray:
    """
    Helper method for calculating Potential of Mean Force (PMF) from Φ(A,x). Supports input and output transforms

        PMF(x) = -KbT * ln(Peq(x)) = 2 * KbT * ln(Φ(A,x))

    {@inheritdoc phi_scaled}
    :return: Potential of Mean Force (PMF) after scaling
    """

    _phi_ax = phi_scaled(x=x,
                         kb_t=kb_t,
                         ks=ks,
                         depth=depth,
                         bias=bias,
                         x_offset=x_offset,
                         x_scale=x_scale,
                         phi_offset=phi_offset,
                         phi_scale=phi_scale)

    return 2 * kb_t * np.log(_phi_ax)


def minimize_double_well_pmf(x_start: float, x_stop: float,
                             ret_min_value: bool,
                             kb_t: float,
                             ks: float,
                             depth: float,
                             bias: float,
                             x_offset: float = 0,
                             x_scale: float = 1,
                             phi_offset: float = 0,
                             phi_scale: float = 1):
    """
    Minimizes double-well pmf within (x_start, x_stop)
    Returns the minima_x and pmf at minima as a tuple (min_x, min_pmf)
    """

    def _func(x):
        return double_well_pmf_scaled(x=x,
                                      kb_t=kb_t,
                                      ks=ks,
                                      depth=depth,
                                      bias=bias,
                                      x_offset=x_offset,
                                      x_scale=x_scale,
                                      phi_offset=phi_offset,
                                      phi_scale=phi_scale)

    return minimize_func(_func, x_start=x_start, x_stop=x_stop, ret_min_value=ret_min_value)


# Utility Functions -----------------------------------------------------

def get_pmf_min_max(x: np.ndarray, pmf: np.ndarray):
    max_i = np.argmax(pmf)

    len_half = len(pmf) // 2
    min_i_left = np.argmin(pmf[:len_half])
    min_i_right = len_half + np.argmin(pmf[len_half:])

    return {
        "minima_left": (x[min_i_left], pmf[min_i_left]),
        "minima_right": (x[min_i_right], pmf[min_i_right]),
        "maxima": (x[max_i], pmf[max_i])
    }


def analyze_pmf_min_max(pmf_df_file_name: str,
                        x_col_name: str = COL_NAME_X,
                        pmf_col_name: str = COL_NAME_PMF,
                        out_file_name: str = None) -> pd.DataFrame:
    pmf_df = read_csv(pmf_df_file_name)

    x = pmf_df[x_col_name].values
    pmf = pmf_df[pmf_col_name].values

    info = get_pmf_min_max(x=x, pmf=pmf)
    keys = info.keys()

    df: pd.DataFrame = pd.DataFrame(columns=("Location", x_col_name, pmf_col_name))

    for i, key in enumerate(keys):
        df.loc[i] = [key, *info[key]]

    max_pmf = info["maxima"][1]
    min_pmf_left = info["minima_left"][1]
    min_pmf_right = info["minima_right"][1]

    if out_file_name:
        to_csv(df, out_file_name, comments=[
            "--------------- PMF Analysis ----------------",
            f"INPUT pmf_file: \"{pmf_df_file_name}\" | x_col_name: {x_col_name} | pmf_col_name: {pmf_col_name}",
            f"-> Barrier Energy (from minima LEFT): {max_pmf - min_pmf_left} ",
            f"-> Barrier Energy (from minima RIGHT): {max_pmf - min_pmf_right} ",
            "----------------------------------------------"
        ])

    return df


def main_analyze_pmf():
    pmf_df_file_name = "results-theory_sim/sp_app/sp_app-fit-2.2.sim_app_pmf_aligned.csv"
    x_col_name = COL_NAME_EXTENSION
    pmf_col_name = COL_NAME_PMF_RECONSTRUCTED

    out_file_name = "results-theory_sim/sp_app/sim_app_pmf_aligned2.2.pmf_re_min_max.txt"

    analyze_pmf_min_max(pmf_df_file_name,
                        x_col_name=x_col_name,
                        pmf_col_name=pmf_col_name,
                        out_file_name=out_file_name)


def main():
    Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)

    T = 300  # Temperature (K)

    A = -0.44
    bias = 0.0717  # 0: symmetric, 0.0717: un-sym
    ks = 10  # Force constant (kcal/mol/Å**2)

    # Domain (in Å)
    x = np.linspace(10, 30, 100, endpoint=False)  # [start = -1.5, stop = 1] with no scale or offset

    # phi_ax = phi_scaled(x, kb_t=Kb * T, ks=ks, depth=A, bias=bias, x_scale=1 / 10, x_scaled_offset=-19.8)  # for symmetric bistable potential

    phi_ax = phi_scaled(x, kb_t=Kb * T, ks=ks, depth=A, bias=bias, x_scale=1 / 8,
                        x_offset=-22.4)  # For un-symmetric potential

    # PMF from Probability Distribution = -KbT ln(Peq(x)) = 2 * KbT * ln(phi(A,x))
    pmf = 2 * Kb * T * np.log(phi_ax)

    print("\n#X\tPMF")
    for i in range(len(x)):
        print(f"{x[i]}\t{pmf[i]}")

    plt.plot(x, pmf)
    plt.show()


if __name__ == "__main__":
    # main()
    main_analyze_pmf()
