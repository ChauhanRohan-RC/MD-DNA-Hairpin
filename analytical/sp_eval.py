import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp

import scipy

import sp_impl
from double_well_pmf import phi_scaled
from double_well_pmf_fit import load_fit_params

"""
Script to Evaluate quantities implemented in "sp_impl.py" 
"""

## CONSTANTS -----------------------------
Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol

Ks = 10  # Force constant (kcal/mol/Å**2)

# PARAMS
x_a = -1.00  # LEFT Boundary (Å)
x_b = 0.87  # RIGHT Boundary (Å)
x_integration_samples = 100

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

# NOT USING OFFSETS FOR NOW
x_offset, x_scale, phi_offset, phi_scale = 0, 1, 0, 1


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


def _sp_vec(x0: np.ndarray | float, t0: np.ndarray | float):
    return sp_impl.sp_vec(x0=x0, t0=t0,
                          t_start=t_integration_start, t_stop=t_integration_stop, t_samples=t_integration_samples,
                          x_a=x_a, x_b=x_b, x_samples=x_integration_samples,
                          n_max=n_max, cyl_dn_a=cyl_dn_a,
                          kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                          depth=depth, bias=bias,
                          x_offset=x_offset, x_scale=x_scale,
                          phi_offset=phi_offset, phi_scale=phi_scale)


def _sp_final_eq_vec(x0: np.ndarray | float):
    return sp_impl.sp_final_eq_vec(x0=x0,
                                   x_a=x_a, x_b=x_b,
                                   n_max=n_max, cyl_dn_a=cyl_dn_a,
                                   kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                                   depth=depth, bias=bias,
                                   x_offset=x_offset, x_scale=x_scale,
                                   phi_offset=phi_offset, phi_scale=phi_scale)


# MAIN -----------------------------------------------------------------------------

def mp_execute(worker_func, input_arr: np.ndarray, process_count: int) -> np.ndarray:
    sample_count = len(input_arr)

    q, r = divmod(sample_count, process_count)
    chunk_size = q if r == 0 else q + 1

    print(f"Computing in Multiprocess Mode"
          f"\n Target Function: {worker_func.__name__}"
          f"\n -> Total CPU(s): {mp.cpu_count()} | Req Processes: {process_count}"
          f"\n -> Total Sample: {sample_count} | Samples per Process: {chunk_size}")

    time_start = time.time()

    chunks = [input_arr[i: min(i + chunk_size, sample_count)] for i in range(0, sample_count, chunk_size)]

    pool = mp.Pool(processes=process_count)
    res = pool.map(worker_func, chunks)

    time_end = time.time()

    print(f"Time taken: {time_end - time_start:.2f} s")
    return np.concatenate(res)

def cond_prob_integral_x_vs_x0_worker(x0: np.ndarray | float):
    return _cond_prob_integral_x_vec(x0=x0, t0=t_0, t=time_instant_test)


def test_cond_prob_integral_x_vs_x0():
    x0 = np.linspace(x_a, x_b, 100, endpoint=True)
    y = mp_execute(cond_prob_integral_x_vs_x0_worker, x0, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "X0": x0,
        "CP_INT_X": y
    })

    df.to_csv("results-sp_eval/sp_eval-cond_prob_int_x_vs_x0.csv", sep="\t", header=True, index=False,
              index_label=False)

    plt.plot(x0, y, label=f"t: {time_instant_test} s")
    plt.xlabel("X0")
    plt.ylabel("CP_INTx(x0, t0, t)")

    plt.legend(loc="upper right")
    plt.savefig("results-sp_eval/sp_eval-cond_prob_int_x_vs_x0.svg")
    plt.show()


def cond_prob_integral_x_vs_t_worker(t: np.ndarray | float):
    return _cond_prob_integral_x_vec(x0=x_0, t0=t_0, t=t)


def test_cond_prob_integral_x_vs_t():
    t_arr = np.linspace(4.2e-8, 4e-6, 100, endpoint=False)
    y = mp_execute(cond_prob_integral_x_vs_t_worker, t_arr, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "T": t_arr,
        "CP_INT_X": y
    })

    df.to_csv("results-sp_eval/sp_eval-cond_prob_int_x_vs_t.csv", sep="\t", header=True, index=False, index_label=False)

    plt.plot(t_arr, y, label=f"x0: {x_0} A | t0: {t_0} s")
    plt.xlabel("t (s)")
    plt.ylabel("CP_INTx(x0, t0, t)")

    plt.legend(loc="upper right")
    plt.savefig("results-sp_eval/sp_eval-cond_prob_int_x_vs_t.svg")
    plt.show()


def fpt_worker(t: np.ndarray):
    return _first_pass_time_vec(x0=x_0, t0=t_0, t=t)


def test_fpt():
    # NOTE: Time range for first_pass_time distribution is 40.825e-9 - 5e-6

    t_arr = np.linspace(4.2e-8, 4e-6, 100, endpoint=False)
    y = mp_execute(fpt_worker, t_arr, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "T": t_arr,
        "FPT": y
    })

    df.to_csv("results-sp_eval/sp_eval-fpt.csv", sep="\t", header=True, index=False, index_label=False)

    plt.plot(t_arr, y, label="FPT vs t")
    plt.xlabel("Time (s)")
    plt.ylabel("First Passage Time FPT(t)")

    plt.legend(loc="upper right")
    plt.savefig("results-sp_eval/sp_eval-fpt.svg")
    plt.show()


def sp_worker(x0: np.ndarray):
    return _sp_vec(x0=x0, t0=t_0)


def test_sp():
    x0 = np.linspace(x_a, x_b, 100, endpoint=True)
    y = mp_execute(sp_worker, x0, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "X0": x0,
        "SP": y
    })

    df.to_csv("results-sp_eval/sp_eval-sp.csv", sep="\t", header=True, index=False, index_label=False)

    plt.plot(x0, y, label=f"SP at t0={t_0} s")
    plt.xlabel("X0 (A)")
    plt.ylabel("SP(x0)")

    plt.legend(loc="upper right")
    plt.savefig("results-sp_eval/sp_eval-sp.svg")
    plt.show()


def sp_final_eq_worker(x0: np.ndarray):
    return _sp_final_eq_vec(x0=x0)


def test_sp_final_eq():
    x0 = np.linspace(x_a, x_b, 100, endpoint=True)
    y = mp_execute(sp_final_eq_worker, x0, mp.cpu_count() - 1)

    df = pd.DataFrame({
        "X0": x0,
        "SP": y
    })

    df.to_csv("results-sp_final_eq/sp_eval-sp_final_eq.csv", sep="\t", header=True, index=False, index_label=False)

    plt.plot(x0, y, label=f"SP Final Eq")
    plt.xlabel("X0 (A)")
    plt.ylabel("SP(x0)")

    plt.legend(loc="upper right")
    plt.savefig("results-sp_final_eq/sp_eval-sp_final_eq.svg")
    plt.show()


def test_sp_integral__pmf_re(sp_vs_x_file, output_data_file, output_sp_fig_file):
    df = pd.read_csv(sp_vs_x_file, sep=r"\s+", comment="#")
    x = df["X0"]
    y = df["SP"]

    y -= np.min(y)

    # Integral in the denominator = Constant
    c = scipy.integrate.trapezoid(y=y, x=x)

    y2 = np.zeros(len(x), dtype=np.float128)

    for i in range(len(x)):
        _v = scipy.integrate.trapezoid(y=y[i:], x=x[i:])
        y2[i] = _v / c

    df["SP_INT"] = y2
    grad: np.ndarray = np.gradient(y2, x)
    print(f"IS SP gradient +ve: {(grad > 0).sum()}")

    # pmf_re = -grad
    pmf_re = KbT * np.log(-grad)
    df["PMF_RE"] = pmf_re

    df.to_csv(output_data_file, sep="\t", header=True, index=False, index_label=False)

    plt.plot(x, y2, label="SP_INT")
    plt.legend(loc="upper right")

    if output_sp_fig_file:
        plt.savefig(output_sp_fig_file)
    plt.show()


def plot_pmf_re(pmf_vs_x_dat_file, output_fig_file):
    df = pd.read_csv(pmf_vs_x_dat_file, sep=r"\s+", comment="#")
    x = df["X0"]
    pmf_re = df["PMF_RE"]
    pmf = _pmf(x)

    plt.plot(x, pmf, label="PMF-IM")
    plt.plot(x, pmf_re, label="PMF-RE")

    plt.legend(loc="upper right")
    if output_fig_file:
        plt.savefig(output_fig_file)
    plt.show()


def test_local():
    x = np.linspace(x_a, x_b, 100, endpoint=True)

    # y = _pmf(x)
    # plt.plot(x, y, label="PMF")

    y = _cond_prob_vec(x, t=time_instant_test, x0=x_0, t0=t_0)

    plt.plot(x, y, label=f"t: {time_instant_test} s, x0: {x_0} A, t0: {t_0} s")
    plt.xlabel("x (A)")
    plt.ylabel("P(x, t, x0, t0)")

    plt.legend(loc="upper right")
    plt.savefig("results-sp_eval/sp_eval-cond_prob.svg")
    plt.show()


if __name__ == '__main__':
    # test_local()
    # test_cond_prob_integral_x_vs_x0()
    # test_cond_prob_integral_x_vs_t()
    # test_fpt()
    # test_sp()

    test_sp_final_eq()

    # test_sp_integral__pmf_re("results-sp_eval/sp_eval-sp.csv",
    #                          "results-sp_eval/sp_eval-sp.csv",
    #                          "results-sp_eval/sp_eval-sp_integral.pdf")

    # test_sp_integral__pmf_re("results-sp_final_eq/sp_eval-sp_final_eq.csv",
    #                          "results-sp_final_eq/sp_eval-sp_final_eq.csv",
    #                          "results-sp_final_eq/sp_eval-sp_final_eq.pdf")

    # plot_pmf_re("results-sp_eval/sp_eval-sp.csv",
    #             "results-sp_eval/sp_eval-pmf_re.pdf")

    # plot_pmf_re("results-sp_final_eq/sp_eval-sp_final_eq.csv",
    #             "results-sp_final_eq/sp_eval-pmf_re.pdf")
