import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import sp_impl
from double_well_pmf import phi_scaled
from double_well_pmf_fit import load_fit_params
from sp_impl import mp_execute

"""
Script to Evaluate quantities implemented in "sp_impl.py" 
"""

## CONSTANTS -----------------------------
Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol

Ks = 10  # Force constant (kcal/mol/Å**2)

# PARAMS -------------------
# NOTE: Optimal x_a and x_b
# -> Without any offset and scales:
#       x_a = -1.0, x_b = 0.87
# -> With T4-DNA hairpin first_double_well "fit-params-1.txt"
#      minima  14.963448853268662, 24.210883674081007
#      x_a = 15.08, x_b = 24.24   or
# -> With T4-DNA hairpin second_double_well "fit-params-2.2.txt"
#      minima 26.637210420334856, 38.45796438483576

x_a = 14.97  # LEFT Boundary (Å) -1.0
x_b = 24.20 # RIGHT Boundary (Å) 0.87
x_integration_samples = 100
x_integration_samples_sp_final_eq = 10000

t_0 = 0
x_0 = x_a

time_instant_test = 1e-4

t_integration_start = t_0
t_integration_stop = 1e-4
t_integration_samples = 200

n_max = 10
cyl_dn_a = 10  # "a" param of cylindrical function
friction_coeff = 1e-7  # friction coeff (eta_1) (in kcal.sec/mol/Å**2). In range (0.5 - 2.38) x 10-7

d1 = KbT / friction_coeff  # diffusion coefficient with beta=1     (in Å**2/s)
print(f"D1: {d1} Å**2/s")

fit_params_file = "results-double_well_fit/fit_params-1.txt"
pmf_fit_params = load_fit_params(fit_params_file)
depth, bias, x_offset, x_scale, phi_offset, phi_scale = pmf_fit_params


## NOT USING OFFSETS FOR NOW
# x_offset, x_scale  = 0, 1
# phi_offset, phi_scale = 0, 1


## Wrappers -----------------------------------------------------------------

def _phi(x: np.ndarray | float):
    return phi_scaled(x, kb_t=KbT, ks=Ks,
                      depth=depth, bias=bias,
                      x_offset=x_offset, x_scale=x_scale,
                      phi_offset=phi_offset, phi_scale=phi_scale)


def _pmf(x: np.ndarray | float):
    return 2 * KbT * np.log(_phi(x))


def _cond_prob_vec(x: np.ndarray | float, t: np.ndarray | float,
                   x0: np.ndarray | float = x_0, t0: np.ndarray | float = t_0):
    return sp_impl.cond_prob_vec(x, t, x0=x0, t0=t0,
                                 n_max=n_max, cyl_dn_a=cyl_dn_a,
                                 kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                                 depth=depth, bias=bias,
                                 x_offset=x_offset, x_scale=x_scale,
                                 phi_offset=phi_offset, phi_scale=phi_scale)


def _cond_prob_integral_x_vec(x0: np.ndarray | float, t0: np.ndarray | float, t: np.ndarray | float):
    return sp_impl.cond_prob_integral_x_vec(x0=x0, t0=t0, t=t,
                                            x_a=x_a, x_b=x_b, x_samples=x_integration_samples,
                                            n_max=n_max, cyl_dn_a=cyl_dn_a,
                                            kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                                            depth=depth, bias=bias,
                                            x_offset=x_offset, x_scale=x_scale,
                                            phi_offset=phi_offset, phi_scale=phi_scale)


def _first_pass_time_vec(x0: np.ndarray | float, t0: np.ndarray | float, t: np.ndarray | float):
    return sp_impl.first_pass_time_vec(x0=x0, t0=t0, t=t,
                                       x_a=x_a, x_b=x_b, x_samples=x_integration_samples,
                                       n_max=n_max, cyl_dn_a=cyl_dn_a,
                                       kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                                       depth=depth, bias=bias,
                                       x_offset=x_offset, x_scale=x_scale,
                                       phi_offset=phi_offset, phi_scale=phi_scale)


def _sp_first_principle(x_a: float = x_a, x_b: float = x_b,
                        x_integration_samples: int = x_integration_samples_sp_final_eq,
                        process_count: int = mp.cpu_count() - 1,
                        return_integrand: bool = True):
    return sp_impl.sp_first_principle(x_a=x_a, x_b=x_b, x_integration_samples=x_integration_samples,
                                      t0=t_0, t_start=t_integration_start, t_stop=t_integration_stop,
                                      t_samples=t_integration_samples,
                                      process_count=process_count,
                                      return_integrand=return_integrand,
                                      n_max=n_max, cyl_dn_a=cyl_dn_a,
                                      kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                                      depth=depth, bias=bias,
                                      x_offset=x_offset, x_scale=x_scale,
                                      phi_offset=phi_offset, phi_scale=phi_scale)


def _sp_final_eq(x_a: float = x_a, x_b: float = x_b,
                 x_integration_samples: int = x_integration_samples_sp_final_eq,
                 process_count: int = mp.cpu_count() - 1,
                 return_integrand: bool = True):
    return sp_impl.sp_final_eq(x_a=x_a, x_b=x_b,
                               x_integration_samples=x_integration_samples,
                               process_count=process_count,
                               return_integrand=return_integrand,
                               n_max=n_max, cyl_dn_a=cyl_dn_a,
                               kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                               depth=depth, bias=bias,
                               x_offset=x_offset, x_scale=x_scale,
                               phi_offset=phi_offset, phi_scale=phi_scale)


def _sp_apparent(x_a: float = x_a, x_b: float = x_b,
                 x_integration_samples: int = x_integration_samples_sp_final_eq,
                 process_count: int = mp.cpu_count() - 1,
                 return_integrand: bool = True):
    return sp_impl.sp_apparent(x_a=x_a, x_b=x_b,
                               x_integration_samples=x_integration_samples,
                               process_count=process_count,
                               return_integrand=return_integrand,
                               kb_t=KbT, ks=Ks,
                               depth=depth, bias=bias,
                               x_offset=x_offset, x_scale=x_scale,
                               phi_offset=phi_offset, phi_scale=phi_scale)


# def _sp_final_eq_vec(x0: np.ndarray | float):
#     return sp_impl.sp_final_eq_vec(x0=x0,
#                                    x_a=x_a, x_b=x_b,
#                                    n_max=n_max, cyl_dn_a=cyl_dn_a,
#                                    kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
#                                    depth=depth, bias=bias,
#                                    x_offset=x_offset, x_scale=x_scale,
#                                    phi_offset=phi_offset, phi_scale=phi_scale)


# MAIN -----------------------------------------------------------------------------

def cond_prob_integral_x_vs_x0_worker(x0: np.ndarray | float):
    return _cond_prob_integral_x_vec(x0=x0, t0=t_0, t=time_instant_test)


def cal_cond_prob_integral_x_vs_x0(out_data_file="results-sp_first_princ/cond_prob_int_x_vs_x0.csv",
                                   out_fig_file="results-sp_first_princ/cond_prob_int_x_vs_x0.pdf"):
    x0 = np.linspace(x_a, x_b, 100, endpoint=True)
    y = mp_execute(cond_prob_integral_x_vs_x0_worker, x0, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "X0": x0,
        "CP_INT_X": y
    })

    if out_data_file:
        df.to_csv(out_data_file, sep="\t", header=True, index=False,
                  index_label=False)

    plt.plot(x0, y, label=f"t: {time_instant_test} s")
    plt.xlabel("X0")
    plt.ylabel("CP_INTx(x0, t0, t)")

    plt.legend(loc="upper right")
    if out_fig_file:
        plt.savefig(out_fig_file)
    plt.show()


def cond_prob_integral_x_vs_t_worker(t: np.ndarray | float):
    return _cond_prob_integral_x_vec(x0=x_0, t0=t_0, t=t)


def cal_cond_prob_integral_x_vs_t(out_data_file="results-sp_first_princ/cond_prob_int_x_vs_t.csv",
                                  out_fig_file="results-sp_first_princ/cond_prob_int_x_vs_t.pdf"):
    t_arr = np.linspace(4.2e-8, 4e-6, 100, endpoint=False)
    y = mp_execute(cond_prob_integral_x_vs_t_worker, t_arr, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "T": t_arr,
        "CP_INT_X": y
    })

    if out_data_file:
        df.to_csv(out_data_file, sep="\t", header=True, index=False, index_label=False)

    plt.plot(t_arr, y, label=f"x0: {x_0} A | t0: {t_0} s")
    plt.xlabel("t (s)")
    plt.ylabel("CP_INTx(x0, t0, t)")

    plt.legend(loc="upper right")
    if out_fig_file:
        plt.savefig(out_fig_file)
    plt.show()


def fpt_worker(t: np.ndarray):
    return _first_pass_time_vec(x0=x_0, t0=t_0, t=t)


# First passage time
def cal_fpt(out_data_file="results-sp_first_princ/fpt_vs_t.csv",
            out_fig_file="results-sp_first_princ/fpt_vs_t.pdf"):
    # NOTE: Time range for first_pass_time distribution is 40.825e-9 - 5e-6

    t_arr = np.linspace(4.2e-8, 4e-6, 100, endpoint=False)
    y = mp_execute(fpt_worker, t_arr, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "T": t_arr,
        "FPT": y
    })

    if out_data_file:
        df.to_csv(out_data_file, sep="\t", header=True, index=False, index_label=False)

    plt.plot(t_arr, y, label="FPT vs t")
    plt.xlabel("Time (s)")
    plt.ylabel("First Passage Time FPT(t)")

    plt.legend(loc="upper right")
    if out_fig_file:
        plt.savefig(out_fig_file)
    plt.show()


def __handle_sp_data(x0: np.ndarray, sp_integrand: np.ndarray | None, sp: np.ndarray,
                     sp_plot_title: str, pmf_plot_title: str,
                     out_data_file, out_fig_file):
    pmf_im = _pmf(x0)  # Imposed PMF
    pmf_re = sp_impl.pmf_re(x=x0, sp=sp, kb_t=KbT)  # Reconstructed PMF

    df = pd.DataFrame({
        "X0": x0,
        "PMF_IM": pmf_im
    })

    if sp_integrand is not None:
        df["SP_INTEGRAND"] = sp_integrand

    df["SP"] = sp
    df["PMF_RE"] = pmf_re

    if out_data_file:
        df.to_csv(out_data_file, sep="\t", header=True, index=False, index_label=False)

    w, h = figaspect(9 / 17)
    fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
    fig.tight_layout(pad=5.0)

    # axes[0].plot(x, sp_integrand, label=f"SP INTEGRAND")
    axes[0].plot(x0, sp_integrand, label=f"SP-INTEGRAND")
    axes[0].plot(x0, sp, label=f"SP")
    axes[0].set_title(sp_plot_title)
    axes[0].set_xlabel("x0 (Å)")
    axes[0].set_ylabel("Sp(x0)")
    axes[0].legend(loc="upper right")

    axes[1].plot(x0, pmf_im, label=f"PMF IMPOSED")
    axes[1].plot(x0, pmf_re, label=f"PMF RECONS")
    axes[1].set_title(pmf_plot_title)
    axes[1].set_xlabel("x0 (Å)")
    axes[1].set_ylabel("PMF(x0) (kcal/mol)")
    axes[1].legend(loc="upper right")

    if out_fig_file:
        plt.savefig(out_fig_file)
    plt.show()


def cal_sp_first_principle(out_data_file="results-sp_first_princ/sp_first_princ.csv",
                           out_fig_file="results-sp_first_princ/sp_first_princ.pdf"):
    x, sp_integrand, sp = _sp_first_principle(x_a=x_a, x_b=x_b,
                                              x_integration_samples=x_integration_samples,
                                              process_count=mp.cpu_count() - 1,
                                              return_integrand=True)

    __handle_sp_data(x0=x, sp_integrand=sp_integrand, sp=sp,
                     sp_plot_title="Sp(x) Theoretical First-Principle",
                     pmf_plot_title="PMF Theoretical First-Principle",
                     out_data_file=out_data_file, out_fig_file=out_fig_file)


def cal_sp_final_eq(out_data_file="results-sp_final_eq/sp_final_eq.csv",
                    out_fig_file="results-sp_final_eq/sp_final_eq.pdf"):
    x, sp_integrand, sp = _sp_final_eq(x_a=x_a, x_b=x_b,
                                       x_integration_samples=x_integration_samples_sp_final_eq,
                                       process_count=mp.cpu_count() - 1,
                                       return_integrand=True)

    __handle_sp_data(x0=x, sp_integrand=sp_integrand, sp=sp,
                     sp_plot_title="Sp(x) Theoretical Final-Eq (EXACT)",
                     pmf_plot_title="PMF Theoretical Final-Eq (EXACT)",
                     out_data_file=out_data_file, out_fig_file=out_fig_file)


def cal_sp_apparent(out_data_file="results-sp_app/sp_app.csv",
                    out_fig_file="results-sp_app/sp_app.pdf"):
    x, sp_integrand, sp = _sp_apparent(x_a=x_a, x_b=x_b,
                                       x_integration_samples=x_integration_samples_sp_final_eq,
                                       process_count=mp.cpu_count() - 1,
                                       return_integrand=True)

    # Just camouflage
    __handle_sp_data(x0=x, sp_integrand=sp_integrand, sp=sp,
                     sp_plot_title="Sp(x) Theoretical Final-Eq (EXACT-APP)",
                     pmf_plot_title="PMF Theoretical Final-Eq (EXACT-APP)",
                     out_data_file=out_data_file, out_fig_file=out_fig_file)


def plot_pmf_re(pmf_vs_x_dat_file, output_fig_file):
    df = pd.read_csv(pmf_vs_x_dat_file, sep=r"\s+", comment="#")
    x = df["X0"]
    pmf_re = df["PMF_RE"]
    pmf = _pmf(x)

    plt.plot(x, pmf, label="PMF-IM")
    plt.plot(x, pmf_re, label="PMF-RE")

    plt.xlabel("x (Å)")
    plt.ylabel("PMF(x) (kcal/mol)")
    plt.legend(loc="upper right")
    if output_fig_file:
        plt.savefig(output_fig_file)
    plt.show()


def plot_pmf_im():
    x = np.linspace(x_a, x_b, 100, endpoint=True)
    y = _pmf(x)

    plt.plot(x, y, label="PMF-IM")
    plt.xlabel("x (Å)")
    plt.ylabel("PMF(x) (kcal/mol)")

    plt.legend(loc="upper right")
    plt.show()


def plot_cond_prob():
    x = np.linspace(x_a, x_b, 100, endpoint=True)
    y = _cond_prob_vec(x, t=time_instant_test, x0=x_0, t0=t_0)

    plt.plot(x, y, label=f"t: {time_instant_test} s, x0: {x_0} A, t0: {t_0} s")
    plt.xlabel("x (Å)")
    plt.ylabel("P(x, t, x0, t0)")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    ## General Tests ------------------------
    # plot_pmf_im()
    # plot_cond_prob()

    ## First Principle  ----------------------
    # cal_cond_prob_integral_x_vs_x0()
    # cal_cond_prob_integral_x_vs_t()
    # cal_fpt()
    # cal_sp_first_principle()

    ## Final Eq ------------------------------
    # TODO: vary x_a and x_b somehow for fit-1 and fit-2.2
    cal_sp_final_eq(out_data_file="results-sp_final_eq/sp_final_eq-fit-1.csv",
                    out_fig_file="results-sp_final_eq/sp_final_eq-fit-1.pdf")

    ## Apparent PMF ------------------------------
    # cal_sp_apparent(out_data_file="results-sp_app/sp_app-fit-1.csv",
    #                 out_fig_file="results-sp_app/sp_app-fit-1.pdf")
