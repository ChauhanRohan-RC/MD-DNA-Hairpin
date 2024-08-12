import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.figure import figaspect

import sp_impl
from double_well_pmf import phi_scaled
from double_well_pmf_fit import load_fit_params
from sp_impl import DEFAULT_PROCESS_COUNT, COL_NAME_X, COL_NAME_PMF_IM, COL_NAME_PMF_RE, COL_NAME_SP

"""
Script to Evaluate quantities implemented in "sp_impl.py"

1. Calculates theoretical Splitting Probability Sp(x) and Reconstructs the-PMF using
    first-principles, final_exact_eq and apparent_pmf
    
2. Plot Theoretical and Simulation SP(x) and PMF_RECONSTRUCTED + PMF_IMPOSED

search for TODO and set the required files/params 
"""

## CONSTANTS -----------------------------
Kb = 1.9872036e-3  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
T = 300  # Temperature (K)
KbT = Kb * T  # kcal/mol
Ks = 10  # Force constant (kcal/mol/Å**2)
friction_coeff = 1e-7  # friction coeff (eta_1) (in kcal.sec/mol/Å**2). In range (0.5 - 2.38) x 10-7
n_max = 10
cyl_dn_a = 10  # "a" param of cylindrical function

fit_params_file = "results-double_well_fit/fit_params-1.2.txt"  # TODO: set fit-params
pmf_fit_params = load_fit_params(fit_params_file)
depth, bias, x_offset, x_scale, phi_offset, phi_scale = pmf_fit_params

## NOT USING OFFSETS FOR NOW
# x_offset, x_scale  = 0, 1
# phi_offset, phi_scale = 0, 1

# PARAMS -------------------
# NOTE: Optimal x_a and x_b
# -> Without any offset and scales:
#       x_a = -1.0, x_b = 0.87
# -> With T4-DNA hairpin
#       * "fit-params-1.1.txt"
#           minima:  14.963448853268662, 24.210883674081007
#           Optimal: x_a = 14.98, x_b = 24.19
#       * "fit-params-1.2.txt"
#           minima:  14.988497234334494, 24.25070294049196
#           Optimal: x_a = 15.0, x_b = 24.23
#       * "fit-params-2.1.txt"
#           minima: 26.887663450991564, 38.00000736802938
#       * "fit-params-2.2.txt"
#           minima: 26.637210420334856, 38.45796438483576
#           Optimal: x_a = 26.65, x_b = 38.41

x_a = 15.0  # TODO: LEFT Boundary (Å)
x_b = 24.23  # TODO: RIGHT Boundary (Å)
x_integration_samples = 100
x_integration_samples_sp_final_eq = 100000  # TODO: set integration sample count

t_0 = 0  # Initial time
x_0 = x_a  # Initially at left well

time_instant_test = 1e-4  # time instant to calculate conditional probability at

t_integration_start = t_0
t_integration_stop = 1e-4
t_integration_samples = 200

d1 = KbT / friction_coeff  # diffusion coefficient with beta=1     (in Å**2/s)
print(f"D1: {d1} Å**2/s")


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


# ==========================================================================================
# ----------------------------  SPLITTING PROBABILITY Wrappers  ---------------------------
# ==========================================================================================

def sp_first_principle(out_data_file: str | None,
                       x_a: float = x_a, x_b: float = x_b,
                       x_integration_samples: int = x_integration_samples_sp_final_eq,
                       process_count: int = DEFAULT_PROCESS_COUNT,
                       return_sp_integrand: bool = True,
                       reconstruct_pmf: bool = True) -> pd.DataFrame:
    return sp_impl.sp_first_principle(x_a=x_a, x_b=x_b, x_integration_samples=x_integration_samples,
                                      t0=t_0, t_start=t_integration_start, t_stop=t_integration_stop,
                                      t_samples=t_integration_samples,
                                      process_count=process_count,
                                      return_sp_integrand=return_sp_integrand,
                                      reconstruct_pmf=reconstruct_pmf,
                                      out_data_file=out_data_file,
                                      n_max=n_max, cyl_dn_a=cyl_dn_a,
                                      kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                                      depth=depth, bias=bias,
                                      x_offset=x_offset, x_scale=x_scale,
                                      phi_offset=phi_offset, phi_scale=phi_scale)


def sp_final_eq(out_data_file: str | None,
                x_a: float = x_a, x_b: float = x_b,
                x_integration_samples: int = x_integration_samples_sp_final_eq,
                process_count: int = DEFAULT_PROCESS_COUNT,
                return_sp_integrand: bool = True,
                reconstruct_pmf: bool = True) -> pd.DataFrame:
    return sp_impl.sp_final_eq(x_a=x_a, x_b=x_b,
                               x_integration_samples=x_integration_samples,
                               process_count=process_count,
                               return_sp_integrand=return_sp_integrand,
                               reconstruct_pmf=reconstruct_pmf,
                               out_data_file=out_data_file,
                               n_max=n_max, cyl_dn_a=cyl_dn_a,
                               kb_t=KbT, ks=Ks, friction_coeff=friction_coeff,
                               depth=depth, bias=bias,
                               x_offset=x_offset, x_scale=x_scale,
                               phi_offset=phi_offset, phi_scale=phi_scale)


def sp_apparent(out_data_file: str | None,
                x_a: float = x_a, x_b: float = x_b,
                x_integration_samples: int = x_integration_samples_sp_final_eq,
                process_count: int = DEFAULT_PROCESS_COUNT,
                return_sp_integrand: bool = True,
                reconstruct_pmf: bool = True) -> pd.DataFrame:
    return sp_impl.sp_apparent(x_a=x_a, x_b=x_b,
                               x_integration_samples=x_integration_samples,
                               process_count=process_count,
                               return_sp_integrand=return_sp_integrand,
                               reconstruct_pmf=reconstruct_pmf,
                               out_data_file=out_data_file,
                               kb_t=KbT, ks=Ks,
                               depth=depth, bias=bias,
                               x_offset=x_offset, x_scale=x_scale,
                               phi_offset=phi_offset, phi_scale=phi_scale)


# MAIN -----------------------------------------------------------------------------

def cond_prob_integral_x_vs_x0_worker(x0: np.ndarray | float):
    return _cond_prob_integral_x_vec(x0=x0, t0=t_0, t=time_instant_test)


def cal_cond_prob_integral_x_vs_x0(out_data_file="results-sp_first_princ/cond_prob_int_x_vs_x0.csv",
                                   out_fig_file="results-sp_first_princ/cond_prob_int_x_vs_x0.pdf"):
    x0 = np.linspace(x_a, x_b, 100, endpoint=True)
    y = sp_impl.mp_execute(cond_prob_integral_x_vs_x0_worker, x0, DEFAULT_PROCESS_COUNT)

    df = pd.DataFrame({
        "X0": x0,
        "CP_INT_X": y
    })

    if out_data_file:
        sp_impl.to_csv(df, out_data_file)

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
    y = sp_impl.mp_execute(cond_prob_integral_x_vs_t_worker, t_arr, DEFAULT_PROCESS_COUNT)

    df = pd.DataFrame({
        "T": t_arr,
        "CP_INT_X": y
    })

    if out_data_file:
        sp_impl.to_csv(df, out_data_file)

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
    y = sp_impl.mp_execute(fpt_worker, t_arr, DEFAULT_PROCESS_COUNT)

    df = pd.DataFrame({
        "T": t_arr,
        "FPT": y
    })

    if out_data_file:
        sp_impl.to_csv(df, out_data_file)

    plt.plot(t_arr, y, label="FPT vs t")
    plt.xlabel("Time (s)")
    plt.ylabel("First Passage Time FPT(t)")

    plt.legend(loc="upper right")
    if out_fig_file:
        plt.savefig(out_fig_file)
    plt.show()


def plot_sp_theory_sim(sp_theory_df: pd.DataFrame,
                       sp_sim_df: pd.DataFrame | None,
                       out_file_name_prefix: str | None,
                       out_fig_file: str | None = "",
                       plot_pmf_im: bool = True,
                       pmf_im_x_extra_left: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                       pmf_im_x_extra_right: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                       sim_data_col_x: str = "EXT_BIN_MED",
                       interp_sim_sp: bool = True,
                       interp_sim_pmf_re: bool = True,
                       interp_sim_samples: int = 200,
                       interp_sim_x_extra_left: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                       interp_sim_x_extra_right: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                       sp_plot_title: str = "Splitting Probability (fold)",
                       pmf_plot_title: str = "PMF",
                       sp_theory_label: str = "Sp (Theory)",
                       sp_sim_label: str = "Sp (Simulation-Traj)",
                       sp_sim_interpolated_label: str = "Sp (Simulation-Traj-Interp)",
                       pmf_im_label: str = "PMF-Imposed (Theory)",
                       pmf_re_theory_label: str = "PMF-Recons (Theory)",
                       pmf_re_sim_label: str = "PMF-Recons (Simulation-Traj)",
                       pmf_re_sim_interpolated_label: str = "PMF-Recons (Simulation-Traj-Interp)"):
    """
    Plots the Splitting Probabilities (SP) and reconstructed PMF(s) from Theory and Simulation Trajectory data
    on the same plot

    -> Imposed PMF is sampled and saved to a file.
    -> Simulation SP and reconstructed PMF are interpolated and the interpolated samples are saved to files

    :param sp_theory_df: pandas DataFrame with theoretical Splitting Probability and Reconstructed PMF
                        i.e. X, SP and PMF_RE columns
                        see "sp_impl.py" methods that save this data

    :param sp_sim_df: pandas DataFrame with simulation data: X, Splitting Probability (SP)
                        and Reconstructed PMF (PMF_RE)

    :param plot_pmf_im: whether to sample and plot Imposed-PMF
    :param pmf_im_x_extra: extra length (in Angstrom) for sampling IMPOSED-PMF.

    :param out_file_name_prefix: (optional) prefix for output file names, like for saving newly created imposed PMF samples,
                                    interpolated simulation SP and reconstructed PMF.
                                    Set to "" or None to disable saving anything.

    :param out_fig_file: (optional) file to save the plot. If not specified, "{out_file_name_prefix}.pdf" is used

    :param sim_data_col_x: column in "sim_data_df" to use as X (reaction coordinate)
    :param interp_sim_sp: whether to interpolate simulation SP samples
    :param interp_sim_pmf_re: whether to interpolate simulation reconstructed-PMF samples
    :param interp_sim_samples: number of samples for interpolation of simulation data
    """

    x = sp_theory_df[COL_NAME_X].values
    sp_theory = sp_theory_df[COL_NAME_SP].values
    pmf_re = sp_theory_df[COL_NAME_PMF_RE].values

    pmf_im_x = None
    pmf_im = None
    if plot_pmf_im:
        # Imposed PMF domain
        no_extra_x = pmf_im_x_extra_left == 0 and pmf_im_x_extra_right == 0
        pmf_im_x = x if no_extra_x else np.linspace(x[0] - pmf_im_x_extra_left,
                                                    x[-1] + pmf_im_x_extra_right,
                                                    num=x_integration_samples_sp_final_eq,
                                                    endpoint=True)

        pmf_im = _pmf(pmf_im_x)  # Imposed PMF

        # Save newly created Imposed-PMF samples to a file
        if out_file_name_prefix:
            df_pmf_im = pd.DataFrame({
                COL_NAME_X: pmf_im_x,
                COL_NAME_PMF_IM: pmf_im
            })

            _im_pmf_file_name = f"{out_file_name_prefix}.pmf_im.csv"
            sp_impl.to_csv(df_pmf_im, _im_pmf_file_name)
            print(f"SP_EVAL: Writing Imposed-PMF samples to file \"{_im_pmf_file_name}\"")

    x_sim = None
    sp_sim = None
    pmf_re_sim = None
    x_sim_interp = None
    sp_sim_interp = None
    pmf_re_sim_interp = None
    if sp_sim_df is not None:
        x_sim = sp_sim_df[sim_data_col_x].values
        sp_sim = sp_sim_df[COL_NAME_SP].values
        pmf_re_sim = sp_sim_df[COL_NAME_PMF_RE].values

        if interp_sim_sp or interp_sim_pmf_re:
            if out_file_name_prefix:
                df_sim_interp = pd.DataFrame()

            x_sim_interp = np.linspace(x_sim[0] - interp_sim_x_extra_left, x_sim[-1] + interp_sim_x_extra_right, interp_sim_samples)
            if out_file_name_prefix:
                df_sim_interp[COL_NAME_X] = x_sim_interp

            if interp_sim_sp:
                _sp_interp_func = scipy.interpolate.interp1d(x_sim, sp_sim, kind="quadratic",
                                                             fill_value="extrapolate")
                sp_sim_interp = _sp_interp_func(x_sim_interp)
                if out_file_name_prefix:
                    df_sim_interp[COL_NAME_SP] = sp_sim_interp

            if interp_sim_pmf_re:
                _pmf_interp_func = scipy.interpolate.interp1d(x_sim, pmf_re_sim, kind="quadratic",
                                                              fill_value="extrapolate")
                pmf_re_sim_interp = _pmf_interp_func(x_sim_interp)
                if out_file_name_prefix:
                    df_sim_interp[COL_NAME_PMF_RE] = pmf_re_sim_interp

            if out_file_name_prefix:
                _sim_interp_file = f"{out_file_name_prefix}.sp_sim_interp.csv"
                sp_impl.to_csv(df_sim_interp, _sim_interp_file)
                print(
                    f"SP_EVAL: Writing interpolated Simulation SP and Reconstructed-PMF samples to file \"{_sim_interp_file}\"")

    w, h = figaspect(9 / 17)
    fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
    fig.tight_layout(pad=5.0)

    # axes[0].plot(x, sp_integrand, label=f"SP-INTEGRAND")
    if sp_sim_df is not None:
        axes[0].scatter(x_sim, sp_sim, label=sp_sim_label)
        if sp_sim_interp is not None:
            axes[0].plot(x_sim_interp, sp_sim_interp, "--", label=sp_sim_interpolated_label)
    axes[0].plot(x, sp_theory, label=sp_theory_label)
    axes[0].set_title(sp_plot_title)
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("Sp(x)")
    axes[0].legend(bbox_to_anchor=(0.2, 1.1), fontsize=7)

    if sp_sim_df is not None:
        axes[1].scatter(x_sim, pmf_re_sim, label=pmf_re_sim_label)
        if pmf_re_sim_interp is not None:
            axes[1].plot(x_sim_interp, pmf_re_sim_interp, "--", label=pmf_re_sim_interpolated_label)
    axes[1].plot(x, pmf_re, label=pmf_re_theory_label)
    if plot_pmf_im and not (pmf_im_x is None or pmf_im is None):
        axes[1].plot(pmf_im_x, pmf_im, label=pmf_im_label)
    axes[1].set_title(pmf_plot_title)
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("PMF(x) (kcal/mol)")
    axes[1].legend(bbox_to_anchor=(1.1, 1.1), fontsize=7)

    if not out_fig_file and out_file_name_prefix:
        out_fig_file = f"{out_file_name_prefix}.pdf"

    if out_fig_file:
        plt.savefig(out_fig_file)
        print(f"SP_EVAL: SP and Reconstructed-PMF plot saved to file \"{out_fig_file}\"")

    plt.show()


def plot_pmf_reconstructed(pmf_vs_x_dat_file, output_fig_file):
    df = sp_impl.read_csv(pmf_vs_x_dat_file)
    x = df[COL_NAME_X]
    pmf_re = df[COL_NAME_PMF_RE]
    pmf = _pmf(x)

    plt.plot(x, pmf, label="PMF-IM")
    plt.plot(x, pmf_re, label="PMF-RE")

    plt.xlabel("x (Å)")
    plt.ylabel("PMF(x) (kcal/mol)")
    plt.legend(loc="upper right")
    if output_fig_file:
        plt.savefig(output_fig_file)
    plt.show()


def plot_pmf_imposed():
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
    # TODO: set file_names before running

    ## General Tests ------------------------
    # plot_pmf_im()
    # plot_cond_prob()

    ## ================================ FIRST PRINCIPLES (APPROX) =====================================
    # cal_cond_prob_integral_x_vs_x0()
    # cal_cond_prob_integral_x_vs_t()
    # cal_fpt()

    if 0:
        sp_first_principle(out_data_file="results-sp_first_princ/sp_first_princ-fit-1.csv",
                           reconstruct_pmf=True,
                           process_count=DEFAULT_PROCESS_COUNT)

    ## ======================== FINAL EQUATION (EXACT) ==================================
    if 0:
        sp_final_eq(out_data_file="results-sp_final_eq/sp_final_eq-fit-1.1.csv",
                    reconstruct_pmf=True,
                    process_count=DEFAULT_PROCESS_COUNT)

    ## ======================= FROM APPARENT PMF (EXACT-EQUILIBRIUM) =====================
    if 0:
        sp_apparent(out_data_file="results-sp_app/sp_app-fit-2.2.csv",
                    reconstruct_pmf=True,
                    process_count=DEFAULT_PROCESS_COUNT)

    ## ----------------------------------------------------------------------------------

    ## Plotting Results -> SP and Reconstructed PMF from theory and simulation
    if 1:
        sp_sim_df = sp_impl.read_csv("sp_traj1.2.csv")
        sp_theory_df = sp_impl.read_csv("results-sp_app/sp_app-fit-1.2.csv")

        plot_sp_theory_sim(sp_theory_df=sp_theory_df,
                           sp_sim_df=sp_sim_df,
                           sim_data_col_x="EXT_BIN_MED",
                           out_file_name_prefix="results-sp_app/sp_app-fit-1.2",
                           interp_sim_sp=True,
                           interp_sim_pmf_re=True,
                           interp_sim_x_extra_left=0.7,
                           interp_sim_x_extra_right=0.8,
                           plot_pmf_im=True,
                           pmf_im_x_extra_left=1.3,
                           pmf_im_x_extra_right=1.4)
