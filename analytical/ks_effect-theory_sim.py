import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import sp_impl
from C import *
from double_well_pmf import get_pmf_min_max_indices
from double_well_pmf_fit import load_fit_params, create_double_well_pmf_func, gen_pmf

"""
Script to analyze the Effect of Ks (linker stiffness) on Theoretical Reconstructed PMF
"""

# TODO: can convert free energy units to KBT, but sim-data units must also be converted

tag = "SIMULATION-2.2"  # TODO: tag to define state
pmf_fit_params_file = "data_sim/pmf_fit/sp_traj-2.2.params.txt"  # TODO: set PMF fit-params
sim_df_file = "data_sim/sp_traj2.2.csv"
out_file_name_prefix = "results-theory_sim/ks_effect/sp_app-2.2"  # TODO: Output file name prefix

pmf_x_search_start, pmf_x_search_stop = 15, 50
align_pmf_minimas = True

# x_a = 20  # LEFT Boundary (Å)
# x_b = 32  # RIGHT Boundary (Å)

# x_0 = x_a  # INITIAL Position (Å)
# t_0 = 0  # Initial time
# time_instant = 1  # time instant to calculate first-principle quantities

kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
temp = 300  # Temperature (K)
kb_t = kb * temp  # (kcal/mol)

## TODO: Stiffness of optical trap
ks_arr = np.array([4, 6, 8, 10])


# beta = 1  # Homogeneity coefficient, in range [0, 1] where 1 is fully homogenous (diffusive) media
# friction_coefficient_beta = 1e-7  # friction coeff (eta_beta) (unit: (s^beta) . kcal/mol/Å**2). In range (0.5 - 2.38) x 10-7

# n_max = 10
# cyl_dn_a = 10  # "a" param of cylindrical function

# x_integration_samples_first_princ = 200
# x_integration_samples_final_eq = 1000  # TODO: set integration sample count
# time_integration_start = t_0
# time_integration_stop = 1e-4
# time_integration_samples = 200

def cal_sp_pmf_re(fit_params,
                  kb_t: float,
                  ks_arr: np.ndarray,
                  pmf_x_search_start: float,
                  pmf_x_search_stop: float,
                  align_pmf_minimas: bool,
                  pmf_x_samples: int = 200,
                  pmf_x_extra_left: float = 1,
                  pmf_x_extra_right: float = 1,
                  out_file_name_prefix: str | None = None,
                  out_data_file_tag: str | None = None):
    pmf_funcs = list(map(lambda ks: create_double_well_pmf_func(fit_params, kb_t=kb_t, ks=ks), ks_arr))
    x, pmf_arr, min_max = gen_pmf(pmf_funcs=pmf_funcs,
                                  x_search_start=pmf_x_search_start,
                                  x_search_stop=pmf_x_search_stop,
                                  align_pmf_minimas=align_pmf_minimas,
                                  align_broadest=False,
                                  x_samples=pmf_x_samples,
                                  x_extra_left=pmf_x_extra_left,
                                  x_extra_right=pmf_x_extra_right)

    sim_df = None
    if sim_df_file:
        sim_df = read_csv(sim_df_file)
        sim_df_x = sim_df[COL_NAME_EXT_BIN_MEDIAN]
        sim_df_sp = sim_df[COL_NAME_SP]
        sim_df_pmf_re = sim_df[COL_NAME_PMF_RECONSTRUCTED] - sim_df[COL_NAME_PMF_RECONSTRUCTED].min()

    ## Plotting PMF ------------------------------

    for i, pmf in enumerate(pmf_arr):
        plt.plot(x, pmf, label=f'ks: {ks_arr[i]:.2f}')

    if sim_df is not None:
        plt.scatter(sim_df_x, sim_df_pmf_re, label="PMF_RE (Simulation-Traj)", c="k")

    # ref_x, ref_pmf = x, pmf_arr[0]
    # min_val_left = np.min(ref_pmf[:len(ref_pmf) // 2])
    # min_val_right = np.min(ref_pmf[len(ref_pmf) // 2:])
    # print(f"Min VAL LEFT: {min_val_left}")
    # print(f"Min VAL RIGHT: {min_val_right}")
    #
    # plt.plot(ref_x, np.full(len(ref_x), min_val_left), '--', c='k')
    # plt.plot(ref_x, np.full(len(ref_x), min_val_right), '--', c='k')

    plt.legend(loc='best')
    plt.title("PMF")
    plt.xlabel("x (A)")
    plt.ylabel("PMF Imposed (kcal/mol)")
    plt.savefig(f"{out_file_name_prefix}.pmf_im.svg")
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

    final_pmf_im_df = pd.DataFrame()
    final_pmf_im_df[COL_NAME_X] = x

    final_sp_pmf_re_df = pd.DataFrame()

    comments = [
        "---------------------",
        f"{out_data_file_tag}",
        f"INPUT fit-params => depth: {fit_params[0]}, bias: {fit_params[1]}, x_offset: {fit_params[2]}, x_scale: {fit_params[3]}, phi_offset: {fit_params[4]}, phi_scale: {fit_params[5]}",
        f"INPUT KbT: {kb_t}",
        f"INPUT Ks values: {ks_arr}",
        "-----------------------------------------------"
    ]

    for i, pmf in enumerate(pmf_arr):
        final_pmf_im_df[f"{COL_NAME_PMF_IMPOSED}{i + 1}"] = pmf

        min_left_i, min_right_i, max_i = get_pmf_min_max_indices(pmf)
        sp_df = sp_impl.sp_apparent2(x=x[min_left_i:min_right_i + 1],
                                     pmf=pmf[min_left_i:min_right_i + 1],
                                     kb_t=kb_t,
                                     return_sp_integrand=False,
                                     reconstruct_pmf=True,
                                     out_data_file=None)

        sp_df[COL_NAME_PMF_RECONSTRUCTED] -= sp_df[COL_NAME_PMF_RECONSTRUCTED].min()
        sp_dfs.append(sp_df)

        final_sp_pmf_re_df[f"{COL_NAME_X}{i + 1}"] = sp_df[COL_NAME_X]
        final_sp_pmf_re_df[f"{COL_NAME_SP}{i + 1}"] = sp_df[COL_NAME_SP]
        final_sp_pmf_re_df[f"{COL_NAME_PMF_RECONSTRUCTED}{i + 1}"] = sp_df[COL_NAME_PMF_RECONSTRUCTED]

        sp_half_i = np.searchsorted(-sp_df[COL_NAME_SP].values, -0.5, side="right")
        print(
            f"CALC-SP: SP for params {fit_params[i]}\n => Sp = 0.5 at ({sp_df[COL_NAME_X][sp_half_i]}, {sp_df[COL_NAME_PMF_RECONSTRUCTED][sp_half_i]})")

        series_label = f"ks = {ks_arr[i]}"
        axes[0].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_SP], label=series_label)
        axes[1].plot(sp_df[COL_NAME_X], sp_df[COL_NAME_PMF_RECONSTRUCTED], label=series_label)

    if sim_df is not None:
        axes[0].scatter(sim_df_x, sim_df_sp, label="Simulation-Traj", c="k")
        axes[1].scatter(sim_df_x, sim_df_pmf_re, label="PMF_RE (Simulation-Traj)", c="k")

    # Saving dataframes: Imposed PMF
    comments[0] = " ------------- Effect of Ks: Imposed PMF ----------------"
    to_csv(final_pmf_im_df, f"{out_file_name_prefix}.pmf_im.csv", comments=comments)

    # Saving dataframes: SP_PMF-RE
    comments[0] = " ------------- Effect of Ks: Sp and PMF_RE ----------------"
    to_csv(final_sp_pmf_re_df, f"{out_file_name_prefix}.sp_pmf_re.csv", comments=comments)

    axes[0].set_xlabel("x (A)")
    axes[0].set_ylabel("Sp")
    axes[0].legend(loc='best')

    axes[1].set_xlabel("x (A)")
    axes[1].set_ylabel("PMF-Recons (kcal/mol)")
    # axes[1].legend(loc='best')
    # axes[1].set_ylim([0, 4])

    plt.savefig(f"{out_file_name_prefix}.svg")
    plt.show()


if __name__ == '__main__':
    fit_params = load_fit_params(fit_param_file=pmf_fit_params_file)

    cal_sp_pmf_re(fit_params=fit_params,
                  kb_t=kb_t, ks_arr=ks_arr,
                  pmf_x_search_start=pmf_x_search_start,
                  pmf_x_search_stop=pmf_x_search_stop,
                  align_pmf_minimas=align_pmf_minimas,
                  pmf_x_extra_left=2,
                  pmf_x_extra_right=2,
                  out_file_name_prefix=out_file_name_prefix,
                  out_data_file_tag=f"INPUT TAG: {tag} | param-file: \"{pmf_fit_params_file}\"")
