import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect

import sp_impl
from C import *
from double_well_pmf import get_pmf_min_max_indices
from double_well_pmf_fit import load_fit_params, create_double_well_pmf_func

"""
Script to create demo-bias (effect of changing bias parameter) pmf and Sp samples for figures
"""

tag = "DEMO-BIAS"
main_dir = "results-theory/demo-bias_effect"
pmf_fit_params_file_suffix = ".params.txt"
out_file_name_prefix = "bias"

pmf_x_search_start, pmf_x_search_stop = -30, 30
pmf_x_samples = 1000
align_pmf_minimas = True

# x_a = 20  # LEFT Boundary (Å)
# x_b = 32  # RIGHT Boundary (Å)

# x_0 = x_a  # INITIAL Position (Å)
# t_0 = 0  # Initial time
# time_instant = 1  # time instant to calculate first-principle quantities

kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
temp = 300  # Temperature (K)
kb_t = kb * temp  # (kcal/mol)

## TODO: Stiffness of optical trap, in units of KbT/[x]^2
# -> FIRST value is for IMPOSED-PMF
ks = 5 * kb_t


# beta = 1  # Homogeneity coefficient, in range [0, 1] where 1 is fully homogenous (diffusive) media
# friction_coefficient_beta = 1e-7  # friction coeff (eta_beta) (unit: (s^beta) . kcal/mol/Å**2). In range (0.5 - 2.38) x 10-7

# n_max = 10
# cyl_dn_a = 10  # "a" param of cylindrical function

# x_integration_samples_first_princ = 200
# x_integration_samples_final_eq = 1000  # TODO: set integration sample count
# time_integration_start = t_0
# time_integration_stop = 1e-4
# time_integration_samples = 200


def get_min_max(pmf_func, x_start: float, x_stop: float):
    _range = x_stop - x_start

    min_left = minimize_func(pmf_func,
                             x_start=x_start,
                             x_stop=x_start + (_range / 2),
                             ret_min_value=False)

    min_right = minimize_func(pmf_func,
                              x_start=x_stop - (_range / 2),
                              x_stop=x_stop,
                              ret_min_value=False)

    maxima = maximize_func(pmf_func,
                           x_start=min_left,
                           x_stop=min_right,
                           ret_max_value=False)

    return min_left, min_right, maxima


# PMF Plots --------------------
def gen_pmf(fit_params: list[np.ndarray],
            kb_t: float,
            ks: float,
            x_search_start: float,
            x_search_stop: float,
            align_pmf_minimas: bool,
            x_samples=200,
            x_extra_left: float = 5,
            x_extra_right: float = 5):
    min_left_low, min_right_high = x_search_stop, x_search_start
    min_left_high, min_right_low = x_search_stop, x_search_start

    pmf_funcs = []
    mins_arr = []
    for params in fit_params:
        pmf_func = create_double_well_pmf_func(fit_params=params, kb_t=kb_t, ks=ks)
        pmf_funcs.append(pmf_func)

        _min_left, _min_right, _maxima = get_min_max(pmf_func, x_search_start, x_search_stop)
        mins_arr.append((_min_left, _min_right))

        print(f"\nPMF for params {params} -----------\n"
              f" -> MINIMA LEFT: ({_min_left}, {pmf_func(_min_left)})\n"
              f" -> MINIMA RIGHT: ({_min_right}, {pmf_func(_min_right)})\n"
              f" -> MAXIMA: ({_maxima}, {pmf_func(_maxima)})\n")

        min_left_low, min_right_high = min(min_left_low, _min_left), max(min_right_high, _min_right)
        min_left_high, min_right_low = max(min_left_high, _min_left), min(min_right_low, _min_right)

    x = np.linspace(min_left_low - x_extra_left, min_right_high + x_extra_right, x_samples)
    pmf_arr = []
    for i in range(len(pmf_funcs)):
        pmf_func = pmf_funcs[i]

        if align_pmf_minimas:
            # Scaling X to match minima's and scaling pmf with the same scale as x
            _min_left, _min_right = mins_arr[i]
            scale = (min_right_high - min_left_low) / (_min_right - _min_left)
            pmf = scale * pmf_func((x / scale) + (_min_left * scale - min_left_low))
            pmf -= np.min(pmf)

            # _min_left, _min_right = mins_arr[i]
            # scale = (min_right_high - min_left_low) / (_min_right - _min_left)
            # x_new = x + _min_left
            # # x_new = (x_new) / scale
            # # x_new = x_new - min_left_low
            # pmf = pmf_func(x_new)
            # pmf -= np.min(pmf)

            # Aligning Left minima
            # pmf = pmf_func(x + (_min_left - min_left_low))
            # pmf -= pmf_func(_min_left)
            # pmf -= np.min(pmf)

            # Aligning Right minima
            # pmf = pmf_func(x + (_min_right - min_right_high))
            # pmf -= pmf_func(_min_right)
        else:
            ## Original PMF: without any x or pmf scale
            pmf = pmf_func(x)

        pmf_arr.append(pmf)

    # tuple (x, [pmf_ks1, pmf_ks2...])
    return x, pmf_arr


def cal_sp_pmf_re(fit_params: list[np.ndarray],
                  kb_t: float,
                  ks: float,
                  pmf_x_search_start: float,
                  pmf_x_search_stop: float,
                  align_pmf_minimas: bool,
                  pmf_x_samples=200,
                  out_file_name_prefix: str | None = None,
                  out_data_file_tag: str | None = None):
    x, pmf_arr = gen_pmf(fit_params=fit_params,
                         kb_t=kb_t,
                         ks=ks,
                         x_search_start=pmf_x_search_start,
                         x_search_stop=pmf_x_search_stop,
                         x_samples=pmf_x_samples,
                         align_pmf_minimas=align_pmf_minimas)

    ## Plotting PMF ------------------------------
    for i, pmf in enumerate(pmf_arr):
        plt.plot(x, pmf, label=f'bias = {fit_params[i][1]}')

    # ref_x, ref_pmf = x, pmf_arr[0]
    # min_val_left = np.min(ref_pmf[:len(ref_pmf) // 2])
    # min_val_right = np.min(ref_pmf[len(ref_pmf) // 2:])
    # print(f"Min VAL LEFT: {min_val_left}")
    # print(f"Min VAL RIGHT: {min_val_right}")
    #
    # plt.plot(ref_x, np.full(len(ref_x), min_val_left), '--', c='k')
    # plt.plot(ref_x, np.full(len(ref_x), min_val_right), '--', c='k')

    # plt.legend(loc='best')
    plt.title("PMF")
    # plt.xlim(-25, 25)
    # plt.ylim(-3, 0.4)
    plt.savefig("pmf-all.svg")
    plt.show()
    ## ------------------------------------

    ## Calculating SP from generated Samples
    sp_dfs = []

    w, h = figaspect(9 / 17)
    fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
    fig.tight_layout(pad=5.0)

    axes[0].set_title("Splitting Probability")
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("Sp(x)")

    axes[1].set_title("PMF-RE")
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("PMF (kcal/mol)")

    for i, pmf in enumerate(pmf_arr):
        min_left_i, min_right_i, max_i = get_pmf_min_max_indices(pmf)

        print(
            f"\nCALC-SP: PMF for params {fit_params[i]}---------\n => MINIMA LEFT: ({x[min_left_i]:.4f}, {pmf[min_left_i]:.4f}), MINIMA RIGHT: ({x[min_right_i]:.4f}, {pmf[min_right_i]:.4f}), MAXIMA: ({x[max_i]:.4f}, {pmf[max_i]:.4f})")

        sp_df = sp_impl.sp_apparent2(x=x[min_left_i:min_right_i + 1],
                                     pmf=pmf[min_left_i:min_right_i + 1],
                                     kb_t=kb_t,
                                     return_sp_integrand=False,
                                     reconstruct_pmf=True,
                                     out_data_file=None)

        # sp_df[COL_NAME_PMF_RECONSTRUCTED] -= sp_df[COL_NAME_PMF_RECONSTRUCTED].min()
        sp_dfs.append(sp_df)

        sp_half_i = np.searchsorted(-sp_df[COL_NAME_SP].values, -0.5, side="right")
        print(f"CALC-SP: SP for params {fit_params[i]}\n => Sp = 0.5 at ({sp_df[COL_NAME_X][sp_half_i]}, {sp_df[COL_NAME_PMF_RECONSTRUCTED][sp_half_i]})")

        comments = [
            "---------------------------",
            f"{out_data_file_tag}",
            f"INPUT KbT: {kb_t}",
            f"INPUT Ks: {ks}  ({ks / kb_t:.4f} KbT/[x]^2)",
            f"INPUT fit-params => depth: {fit_params[i][0]}, bias: {fit_params[i][1]}, x_offset: {fit_params[i][2]}, x_scale: {fit_params[i][3]}, phi_offset: {fit_params[i][4]}, phi_scale: {fit_params[i][5]}",
            "-----------------------------------------------"
        ]

        series_label = f"bias = {fit_params[i][1]}"

        pmf_im_df = pd.DataFrame({COL_NAME_X: x,
                                  COL_NAME_PMF_IMPOSED: pmf})

        # Saving Imposed-PMF Dataframe
        comments[0] = "-------- Imposed PMF -----------"
        to_csv(pmf_im_df, f"{out_file_name_prefix}-{i + 1}.pmf_im.csv", comments=comments)

        comments[0] = "------------------ Sp-PMF_RE ----------------"
        to_csv(sp_df, f"{out_file_name_prefix}-{i + 1}.sp_pmf_re.csv", comments=comments)

        axes[0].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_SP], label=series_label)
        axes[1].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_PMF_RECONSTRUCTED], label=series_label)

    # axes[1].set_ylim([0, 4])
    # axis[0].legend(loc='best')
    axes[1].legend(loc='best')

    plt.savefig(f"{out_file_name_prefix}.sp_pmf_re.svg")
    plt.show()


if __name__ == '__main__':
    # print(sp_impl.critical_bias(-0.4))
    print(f"LOG: Working Dir: {main_dir}")

    files = [f for f in map(lambda a: os.path.join(main_dir, a), os.listdir(main_dir)) if os.path.isfile(f) and f.endswith(pmf_fit_params_file_suffix)]
    print(f"LOG: PARAMS file(s): {files}")

    fit_params = [load_fit_params(f) for f in files]

    cal_sp_pmf_re(fit_params=fit_params,
                  kb_t=kb_t, ks=ks,
                  pmf_x_search_start=pmf_x_search_start,
                  pmf_x_search_stop=pmf_x_search_stop,
                  pmf_x_samples=pmf_x_samples,
                  align_pmf_minimas=align_pmf_minimas,
                  out_file_name_prefix=os.path.join(main_dir, out_file_name_prefix),
                  out_data_file_tag=f"INPUT TAG: {tag} | param-file: \"{pmf_fit_params_file_suffix}\"")
