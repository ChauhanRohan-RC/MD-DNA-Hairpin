from C import read_csv, BOLTZMANN_CONST_KCAL_PER_MOL_K, COL_NAME_EXTENSION, COL_NAME_EXT_BIN_MEDIAN, \
    DEFAULT_PROCESS_COUNT
from sp_eval import SpEval

fit_params_file = "results-test/params-test1.txt"  # TODO: set fit-params
# minima: -0.6618808379709121, 0.6618808379709121, optimal: -0.9, 0.9
x_a = 2 + (-0.8/2)  # TODO: LEFT Boundary (Å)
x_b = 2 + (0.8/2)  # TODO: RIGHT Boundary (Å)

x_0 = x_a   # Initially at left well
t_0 = 0     # Initial time
time_instant = 1e-4  # time instant to calculate first-principle quantities

kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
temp = 300  # Temperature (K)
kb_t = kb * temp  # (kcal/mol)
ks = 10  # Force constant of optical-trap (kcal/mol/Å**2)
friction_coefficient = 1e-7  # friction coefficient (eta_1) (in kcal.sec/mol/Å**2). Optimal range (0.5 - 2.38) x 10-7

n_max = 10
cyl_dn_a = 10  # "a" param of cylindrical function

x_integration_samples_first_princ = 100
x_integration_samples_sp_final_eq = 10000  # TODO: set integration sample count
time_integration_start = t_0
time_integration_stop = 1e-4
time_integration_samples = 200

if __name__ == '__main__':

    ## Creating SpEval Instance  ------------------------------------------------------------
    sp_eval = SpEval(x_a=x_a, x_b=x_b,
                     x_0=x_0, t_0=t_0,
                     time_instant=time_instant,
                     n_max=n_max, cyl_dn_a=cyl_dn_a,
                     kb_t=kb_t, ks=ks, friction_coefficient=friction_coefficient,
                     x_integration_samples_sp_first_princ=x_integration_samples_first_princ,
                     x_integration_samples_sp_final_eq=x_integration_samples_sp_final_eq,
                     time_integration_start=time_integration_start,
                     time_integration_stop=time_integration_stop,
                     time_integration_samples=time_integration_samples)

    if fit_params_file:
        sp_eval.load_pmf_fit_params(fit_params_file=fit_params_file)

    ## General Tests -----------------------------------------------------------------
    # print(sp_eval.get_pmf_minima(0.5, 1))
    # sp_eval.plot_pmf_imposed(None, None)
    # sp_eval.plot_cond_prob(None, None)

    ## ================================ FIRST PRINCIPLES (APPROX) =====================================
    # sp_eval.cal_cond_prob_integral_x_vs_x0(out_data_file="results-theory_sim/sp_first_princ/cond_prob_int_x_vs_x0.csv",
    #                                        out_fig_file="results-theory_sim/sp_first_princ/cond_prob_int_x_vs_x0.pdf")
    #
    # sp_eval.cal_cond_prob_integral_x_vs_t(out_data_file="results-theory_sim/sp_first_princ/cond_prob_int_x_vs_t.csv",
    #                                       out_fig_file="results-theory_sim/sp_first_princ/cond_prob_int_x_vs_t.pdf")
    #
    # sp_eval.cal_fpt(out_data_file="results-theory_sim/sp_first_princ/fpt_vs_t.csv",
    #                 out_fig_file="results-theory_sim/sp_first_princ/fpt_vs_t.pdf")

    if 0:
        sp_eval.sp_first_principle(out_data_file="results-test/sp_first_princ/sp_first_princ-1.csv",
                                   reconstruct_pmf=True,
                                   process_count=DEFAULT_PROCESS_COUNT)

    ## ======================== FINAL EQUATION (EXACT) ==================================
    if 1:
        sp_eval.sp_final_eq(out_data_file="results-test/sp_final_eq/sp_final_eq-1.csv",
                            reconstruct_pmf=True,
                            process_count=DEFAULT_PROCESS_COUNT)

    ## ======================= FROM APPARENT PMF (EXACT-EQUILIBRIUM) =====================
    if 0:
        sp_eval.sp_apparent(out_data_file="results-test/sp_app/sp_app-1.csv",
                            reconstruct_pmf=True,
                            process_count=DEFAULT_PROCESS_COUNT)

    ## ----------------------------------------------------------------------------------

    ## Plotting Results -> SP and Reconstructed PMF from theory and simulation
    if 1:
        sim_traj_df = None  # (optional)
        sim_app_pmf_df = None  # (optional)
        sp_theory_df = read_csv("results-test/sp_final_eq/sp_final_eq-1.csv")

        sp_eval.plot_sp_theory_sim(sp_theory_df=sp_theory_df,
                                   sim_traj_df=sim_traj_df,
                                   sim_traj_df_col_x=COL_NAME_EXT_BIN_MEDIAN,
                                   sim_app_pmf_df=sim_app_pmf_df,
                                   sim_app_pmf_df_col_x=COL_NAME_EXTENSION,
                                   out_file_name_prefix="results-test/sp_final_eq/sp_final_eq-1",
                                   align_sim_app_pmf=True,
                                   align_sim_app_pmf_left_half_only=True,
                                   interp_sim_traj_sp=True,
                                   interp_sim_traj_pmf_re=True,
                                   plot_interp_sim_traj_sp=True,
                                   plot_interp_sim_traj_pmf_re=True,
                                   interp_sim_traj_x_extra_left=0.7,
                                   interp_sim_traj_x_extra_right=0.8,
                                   plot_pmf_im=True,
                                   pmf_im_x_extra_left=0.2,
                                   pmf_im_x_extra_right=0.2)
