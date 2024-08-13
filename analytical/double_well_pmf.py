import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
Models Double-Well PMF with depth and bias parameters

THe Double-Well PMF is modelled using the quantity phi(A, B, x) where A is the depth parameter
and B is the bias parameter. It is given by the linear combination of even and odd solution of Weber equation

        Φ(A,x) = y1(A,x) + bias * y2(A,x)

The corresponding Probability Distribution and PMF are given as

        Probability distribution Peq(x) = phi(A,x)^-2
        PMF = -Kb * T * ln(Peq(x)) = 2 * Kb * T * ln(phi(A,x))
"""

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
    _y1: np.ndarray = scipy.special.hyp1f1((depth / 2) + 0.25, 0.5, _x_hyper)

    if bias != 0:
        # Even solution of weber equation
        _y2: np.ndarray = bias * math.sqrt(ks / kb_t) * x * scipy.special.hyp1f1((depth / 2) + 0.75, 1.5, _x_hyper)
    else:
        _y2: np.ndarray = np.zeros(len(x))

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

        x = (x * x_offset) * x_scale
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


def minimize_func(func, x_start: float, x_stop: float):
    """
    Minimizes the given function within (x_start, x_stop)
    Returns the minima_x and value at minima as a tuple (min_x, func(min_x))
    """
    opt_res = scipy.optimize.minimize_scalar(func, method="bounded", bounds=(x_start, x_stop))
    return opt_res.x, opt_res.fun


def minimize_double_well_pmf(x_start: float, x_stop: float,
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

    return minimize_func(_func, x_start=x_start, x_stop=x_stop)


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
    main()
