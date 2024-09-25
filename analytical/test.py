import numpy as np
from matplotlib.figure import figaspect

from C import *
import matplotlib.pyplot as plt


theory_file = "results-theory_sim/sp_first_princ/sp_first_princ-fit-1.2.cond_prob_vs_t.fold.csv"
sim_file = "data_sim/sp_traj1.2.pdf_re.csv"

theory_df = read_csv(theory_file)
sim_df = read_csv(sim_file)

for i in range(1, len(theory_df.columns)):
    plt.plot(theory_df[theory_df.columns[0]], theory_df[theory_df.columns[i]], label=f"COND_PROB_{i - 1}")

plt.plot(sim_df[sim_df.columns[0]], sim_df[sim_df.columns[1]] / 1000, label="SIM")
plt.legend(loc="best")
plt.savefig("colors.pdf")
plt.show()


# kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
# temp = 300  # Temperature (K)
# kb_t = kb * temp  # (kcal/mol)
# # ks = 1 / KCAL_PER_MOL_A2_TO_pN_PER_nM  # Force constant of optical-trap (kcal/mol/Å**2)
# ks = 10  # Force constant of optical-trap (kcal/mol/Å**2)
#
# beta = 1  # Homogeneity coefficient, in range [0, 1] where 1 is fully homogenous (diffusive) media
# # [with KbT = 4.1 pN nm] => 41 will give D1 = 0.1 nm^2/us  |  8.2 will give D1 = 0.5 nm^2/us
# friction_coefficient_beta = (41 / KCAL_PER_MOL_A2_TO_pN_PER_nM) * 1e-6  # friction coeff (eta_beta) (unit: (s^beta) . kcal/mol/Å**2).
# # friction_coefficient = (41 / KCAL_PER_MOL_A2_TO_pN_PER_nM) * 1e-6
#
# n_max = 10
# cyl_dn_a = 10  # "a" param of cylindrical function
#
#
# x_col_name = COL_NAME_X
# sp_col_name = COL_NAME_SP
# pmf_col_name = COL_NAME_PMF
# pmf_re_col_name = COL_NAME_PMF_RECONSTRUCTED
#
#
# exp_pmf = read_csv("data_exp/pmf-1.1.csv")
# exp_sp = read_csv("data_exp/sp-1.1.csv")
# theory_df = read_csv("results-theory_exp/sp_app/sp_app-1.1.csv")
#
# out_fig_file = "results-theory_exp/sp_app/sp_app-1.1.temp.svg"
#
# w, h = figaspect(9 / 17)
# fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
# fig.tight_layout(pad=5.0)
#
# axes[0].scatter(exp_sp[x_col_name][::2], exp_sp[sp_col_name][::2])
# axes[0].plot(theory_df[x_col_name], theory_df[sp_col_name], c='orange')
# axes[0].set_xlabel(r"$x$ ($Å$)")
# axes[0].set_ylabel(r"$Sp(x)$")
#
# axes[1].scatter(exp_pmf[x_col_name], exp_pmf[pmf_col_name] / kb_t)
# axes[1].plot(theory_df[x_col_name], theory_df[pmf_re_col_name] / kb_t, c='orange')
# axes[1].set_xlabel(r"$x$ ($Å$)")
# axes[1].set_ylabel("$U(x)$ ($k_BT$)")
# axes[1].set_ylim([None, -3.5])
#
# if out_fig_file:
#     plt.savefig(out_fig_file)
# plt.show()