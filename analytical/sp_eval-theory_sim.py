from C import read_csv, BOLTZMANN_CONST_KCAL_PER_MOL_K, COL_NAME_EXTENSION, COL_NAME_EXT_BIN_MEDIAN, \
    DEFAULT_PROCESS_COUNT
from sp_eval import SpEval

"""
Script to Evaluate, Compare and Plot 
    
    1. Splitting Probabilities (first_principles, final_exact, apparent_pmf approaches)
    2. PMF_IMPOSED and PMF_RECONSTRUCTED 
    
for THEORY and SIMULATIONS

USAGE: search for "TODO" and set the required params and file_names 
"""

# ---------------------------------- PARAMS --------------------------------------
# NOTE: Optimal x_a and x_b
# -> Without any offset and scales (depth = -0.44, bias 0):
#       minima (ks = 6950 pN/nm): -0.6618808379709121, 0.6618808379709121
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

pmf_fit_params_file = "data_sim/pmf_fit/sp_traj-2.2.params.txt"  # TODO: set PMF fit-params
x_a = 26.65  # TODO: LEFT Boundary (Å)
x_b = 38.41  # TODO: RIGHT Boundary (Å)

x_0 = x_a  # Initially at left well
t_0 = 0  # Initial time
time_instant = 1e-4  # time instant to calculate first-principle quantities

kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
temp = 300  # Temperature (K)
kb_t = kb * temp  # (kcal/mol)
ks = 10  # Force constant of optical-trap (kcal/mol/Å**2)
friction_coefficient = 1e-7  # friction coefficient (eta_1) (in kcal.sec/mol/Å**2). Optimal range (0.5 - 2.38) x 10-7

n_max = 10
cyl_dn_a = 10  # "a" param of cylindrical function

x_integration_samples_first_princ = 100
x_integration_samples_sp_final_eq = 100000  # TODO: set integration sample count
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

    if pmf_fit_params_file:
        sp_eval.load_pmf_fit_params(fit_params_file=pmf_fit_params_file)

    ## General Tests -----------------------------------------------------------------
    # print(sp_eval.get_pmf_minima(0.5, 1))
    # sp_eval.plot_pmf_imposed()
    # sp_eval.plot_cond_prob()

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
        sp_eval.sp_first_principle(out_data_file="results-theory_sim/sp_first_princ/sp_first_princ-fit-1.csv",
                                   reconstruct_pmf=True,
                                   process_count=DEFAULT_PROCESS_COUNT)

    ## ======================== FINAL EQUATION (EXACT) ==================================
    if 0:
        sp_eval.sp_final_eq(out_data_file="results-theory_sim/sp_final_eq/sp_final_eq-fit-1.1.csv",
                            reconstruct_pmf=True,
                            process_count=DEFAULT_PROCESS_COUNT)

    ## ======================= FROM APPARENT PMF (EXACT-EQUILIBRIUM) =====================
    if 1:
        sp_eval.sp_apparent(out_data_file="results-theory_sim/sp_app/sp_app-fit-2.2.csv",
                            reconstruct_pmf=True,
                            process_count=DEFAULT_PROCESS_COUNT)

    ## ----------------------------------------------------------------------------------

    ## Plotting Results -> SP and Reconstructed PMF from theory and simulation
    if 1:
        sim_traj_df = read_csv("data_sim/sp_traj2.2.csv")  # (optional)
        sim_app_pmf_df = read_csv("data_sim/sp_pmf2.csv")  # (optional)
        sp_theory_df = read_csv("results-theory_sim/sp_app/sp_app-fit-2.2.csv")

        sp_eval.plot_sp_theory_sim(sp_theory_df=sp_theory_df,
                                   sim_traj_df=sim_traj_df,
                                   sim_traj_df_col_x=COL_NAME_EXT_BIN_MEDIAN,
                                   sim_app_pmf_df=sim_app_pmf_df,
                                   sim_app_pmf_df_col_x=COL_NAME_EXTENSION,
                                   out_file_name_prefix="results-theory_sim/sp_app/sp_app-fit-2.2",
                                   align_sim_app_pmf=True,
                                   align_sim_app_pmf_left_half_only=True,
                                   interp_sim_traj_sp=True,
                                   interp_sim_traj_pmf_re=True,
                                   plot_interp_sim_traj_sp=True,
                                   plot_interp_sim_traj_pmf_re=True,
                                   interp_sim_traj_x_extra_left=0.7,
                                   interp_sim_traj_x_extra_right=0.8,
                                   plot_pmf_im=True,
                                   pmf_im_x_extra_left=1.3,
                                   pmf_im_x_extra_right=1.4)
