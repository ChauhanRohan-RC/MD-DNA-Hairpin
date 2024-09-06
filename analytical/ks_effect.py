import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import sp_impl
from C import *
from double_well_pmf import get_pmf_min_max_indices
from double_well_pmf_fit import load_fit_params, create_double_well_pmf_func

"""
Script to analyze the Effect of Ks (linker stiffness) on Theoretical Reconstructed PMF
"""

tag = "ASYMMETRIC_PMF-BIAS_LOW"  # TODO: tag to define state
pmf_fit_params_file = "results-theory/ks_effect/asymm-bias_low/asymm-bias_low.params.txt"  # TODO: set PMF fit-params
out_data_file_name_prefix = "results-theory/ks_effect/asymm-bias_low/asymm-bias_low"  # TODO: Output file name prefix

pmf_x_search_start, pmf_x_search_stop = -6, 6
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
ks_arr = np.array([10.3, 4, 6, 8, 10]) * kb_t


# beta = 1  # Homogeneity coefficient, in range [0, 1] where 1 is fully homogenous (diffusive) media
# friction_coefficient_beta = 1e-7  # friction coeff (eta_beta) (unit: (s^beta) . kcal/mol/Å**2). In range (0.5 - 2.38) x 10-7

# n_max = 10
# cyl_dn_a = 10  # "a" param of cylindrical function

# x_integration_samples_first_princ = 200
# x_integration_samples_final_eq = 1000  # TODO: set integration sample count
# time_integration_start = t_0
# time_integration_stop = 1e-4
# time_integration_samples = 200


def get_min_left_right(pmf_func, x_start: float, x_stop: float):
    min_left = minimize_func(pmf_func,
                             x_start=x_start,
                             x_stop=x_start + ((x_stop - x_start) / 2),
                             ret_min_value=False)

    min_right = minimize_func(pmf_func,
                              x_start=x_start + ((x_stop - x_start) / 2),
                              x_stop=x_stop,
                              ret_min_value=False)

    return min_left, min_right


# ---------------------------
# def plot_sp(plot_sp: bool, align_mode: str):  # align_mode: ('barrier', 'minimas')
#     sp_evals = []
#     for ks in ks_arr:
#         min_left, min_right = get_min_left_right(ks, x_start, x_stop)
#
#         sp_ev: SpEval = SpEval(x_a=min_left, x_b=min_right,
#                                x_0=min_left, t_0=t_0,
#                                time_instant=time_instant,
#                                n_max=n_max, cyl_dn_a=cyl_dn_a,
#                                kb_t=kb_t, ks=ks,
#                                beta=beta, friction_coefficient_beta=friction_coefficient_beta,
#                                x_integration_samples_first_princ=x_integration_samples_first_princ,
#                                x_integration_samples_final_eq=x_integration_samples_final_eq,
#                                time_integration_start=time_integration_start,
#                                time_integration_stop=time_integration_stop,
#                                time_integration_samples=time_integration_samples)
#
#         sp_ev.load_pmf_fit_params(fit_params_file=pmf_fit_params_file)
#         sp_evals.append(sp_ev)
#
#     ref_x, ref_pmf = None, None
#     ref_x_min1, ref_x_min2 = None, None
#     sp_dfs = []
#     for sp_ev in sp_evals:
#         out_data_file = f"test_sp_ks-{sp_ev.ks}.csv"
#         sp_df = sp_ev.sp_apparent(out_data_file=out_data_file, reconstruct_pmf=True)
#         sp_dfs.append(sp_df)
#
#         if align_mode == 'barrier':
#             # Aligning Barrier
#             i_max = np.argmax(sp_df[COL_NAME_PMF_RECONSTRUCTED])
#             max_x = sp_df[COL_NAME_X][i_max]
#             max_pmf = sp_df[COL_NAME_PMF_RECONSTRUCTED][i_max]
#             if ref_x is None:
#                 ref_x = max_x
#                 ref_pmf = max_pmf
#             else:
#                 sp_df[COL_NAME_X] += (ref_x - max_x)
#                 sp_df[COL_NAME_PMF_RECONSTRUCTED] += (ref_pmf - max_pmf)
#         elif align_mode == 'minimas':
#             # Aligning MINIMA's
#             x_col = sp_df[COL_NAME_X].values
#             pmf_re_col = sp_df[COL_NAME_PMF_RECONSTRUCTED].values
#             half_len = len(pmf_re_col) // 2
#             i_min_left = np.argmin(pmf_re_col[:half_len])
#             i_min_right = half_len + np.argmin(pmf_re_col[half_len:])
#             x_min_left = x_col[i_min_left]
#             x_min_right = x_col[i_min_right]
#
#             if ref_x_min1 is None:
#                 ref_x_min1 = x_min_left
#                 ref_x_min2 = x_min_right
#                 scale = 1
#             else:
#                 scale = (ref_x_min2 - ref_x_min1) / (x_min_right - x_min_left)
#             sp_df[COL_NAME_X] *= scale
#             sp_df[COL_NAME_PMF_RECONSTRUCTED] = (pmf_re_col - np.min(pmf_re_col)) * scale
#
#         # Plotting
#         plt.plot(sp_df[COL_NAME_X], sp_df[COL_NAME_SP if plot_sp else COL_NAME_PMF_RECONSTRUCTED],
#                  label=f'{sp_ev.ks:g}')
#
#     plt.legend(loc='best')
#     plt.show()


# PMF Plots --------------------
def gen_pmf(fit_params,
            kb_t: float,
            ks_arr: np.ndarray,
            x_search_start: float,
            x_search_stop: float,
            align_pmf_minimas: bool,
            x_samples=200,
            x_extra_left: float = 0.1,
            x_extra_right: float = 0.1):
    min_left_low, min_right_high = x_search_stop, x_search_start
    min_left_high, min_right_low = x_search_stop, x_search_start

    pmf_funcs = []
    mins_arr = []
    for ks in ks_arr:
        pmf_func = create_double_well_pmf_func(fit_params=fit_params, kb_t=kb_t, ks=ks)
        pmf_funcs.append(pmf_func)

        _min_left, _min_right = get_min_left_right(pmf_func, x_search_start, x_search_stop)
        mins_arr.append((_min_left, _min_right))

        print(f"Minima for ks {ks:.4f} => LEFT: {_min_left:.4f}, RIGHT: {_min_right:.4f}")
        min_left_low, min_right_high = min(min_left_low, _min_left), max(min_right_high, _min_right)
        min_left_high, min_right_low = max(min_left_high, _min_left), min(min_right_low, _min_right)

    x = np.linspace(min_left_low - x_extra_left, min_right_high + x_extra_right, x_samples)
    pmf_arr = []

    for i, ks in enumerate(ks_arr):
        pmf_func = pmf_funcs[i]

        if align_pmf_minimas:
            # Scaling X to match minima's and scaling pmf with the same scale as x
            _min_left, _min_right = mins_arr[i]
            scale = (min_right_high - min_left_low) / (_min_right - _min_left)
            pmf = scale * pmf_func(x / scale)
            pmf -= np.min(pmf)
        else:
            ## Original PMF: without any x or pmf scale
            pmf = pmf_func(x)

        pmf_arr.append(pmf)

    # tuple (x, [pmf_ks1, pmf_ks2...])
    return x, pmf_arr


def cal_sp_pmf_re(fit_params,
                  kb_t: float,
                  ks_arr: np.ndarray,
                  pmf_x_search_start: float,
                  pmf_x_search_stop: float,
                  align_pmf_minimas: bool,
                  pmf_x_samples=200,
                  out_file_name_prefix: str | None = None,
                  out_data_file_tag: str | None = None):
    x, pmf_arr = gen_pmf(fit_params=fit_params,
                         kb_t=kb_t,
                         ks_arr=ks_arr,
                         x_search_start=pmf_x_search_start,
                         x_search_stop=pmf_x_search_stop,
                         x_samples=pmf_x_samples,
                         align_pmf_minimas=align_pmf_minimas)

    ## Plotting PMF ------------------------------
    # for i, pmf in enumerate(pmf_arr):
    #     if i == 0:  # Imposed PMF
    #         plt.plot(x, pmf, label=f'Imposed (ks: {ks_arr[0]:.2f})', c='black')
    #     else:
    #         plt.plot(x, pmf, label=f'ks: {ks_arr[i]:.2f}')
    #
    # ref_x, ref_pmf = x, pmf_arr[0]
    # min_val_left = np.min(ref_pmf[:len(ref_pmf) // 2])
    # min_val_right = np.min(ref_pmf[len(ref_pmf) // 2:])
    # print(f"Min VAL LEFT: {min_val_left}")
    # print(f"Min VAL RIGHT: {min_val_right}")
    #
    # plt.plot(ref_x, np.full(len(ref_x), min_val_left), '--', c='k')
    # plt.plot(ref_x, np.full(len(ref_x), min_val_right), '--', c='k')
    #
    # plt.legend(loc='best')
    # plt.title("PMF")
    # plt.show()
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

        sp_df = sp_impl.sp_apparent2(x=x[min_left_i:min_right_i + 1],
                                     pmf=pmf[min_left_i:min_right_i + 1],
                                     kb_t=kb_t,
                                     return_sp_integrand=False,
                                     reconstruct_pmf=True,
                                     out_data_file=None)

        sp_df[COL_NAME_PMF_RECONSTRUCTED] -= sp_df[COL_NAME_PMF_RECONSTRUCTED].min()
        sp_dfs.append(sp_df)

        common_comments = [
            f"{out_data_file_tag}",
            f"INPUT KbT: {kb_t}",
            f"INPUT Ks: {ks_arr[i]}  ({ks_arr[i]/kb_t:.4f} KbT/[x]^2)",
            f"INPUT fit-params => depth: {fit_params[0]}, bias: {fit_params[1]}, x_offset: {fit_params[2]}, x_scale: {fit_params[3]}, phi_offset: {fit_params[4]}, phi_scale: {fit_params[5]}",
            "-----------------------------------------------"
        ]

        series_label = f"ks: {ks_arr[i] / kb_t:g} KbT/[x]^2"
        if i == 0:  # Imposed (Hidden) PMF
            pmf_im_df = pd.DataFrame({COL_NAME_X: sp_df[COL_NAME_X],
                                      COL_NAME_PMF_IMPOSED: sp_df[COL_NAME_PMF_RECONSTRUCTED]})

            # Saving Imposed-PMF Dataframe
            comments = common_comments.copy()
            comments.insert(0, "--------- Imposed PMF (with Highest Ks) for Effect of Ks on PMF_RE -----------")

            to_csv(pmf_im_df, f"{out_file_name_prefix}.pmf_im.csv", comments=comments)

            axes[1].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_PMF_RECONSTRUCTED],
                         label=f"Imposed ({series_label})", c='black')
        else:
            # Saving sp-pmf_re Dataframe
            comments = common_comments.copy()
            comments.insert(0, "------------------ Effect of Ks on Reconstructed-PMF ----------------")

            to_csv(sp_df, f"{out_file_name_prefix}.sp_pmf_re.ks-{i}.csv", comments=comments)

            axes[0].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_SP], label=series_label)
            axes[1].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_PMF_RECONSTRUCTED], label=series_label)

    # axes[1].set_ylim([0, 4])
    # axis[0].legend(loc='best')
    axes[1].legend(loc='best')

    plt.savefig(f"{out_file_name_prefix}.pdf")
    plt.show()


if __name__ == '__main__':
    fit_params = load_fit_params(fit_param_file=pmf_fit_params_file)

    cal_sp_pmf_re(fit_params=fit_params,
                  kb_t=kb_t, ks_arr=ks_arr,
                  pmf_x_search_start=pmf_x_search_start,
                  pmf_x_search_stop=pmf_x_search_stop,
                  align_pmf_minimas=align_pmf_minimas,
                  out_file_name_prefix=out_data_file_name_prefix,
                  out_data_file_tag=f"INPUT TAG: {tag} | param-file: \"{pmf_fit_params_file}\"")
