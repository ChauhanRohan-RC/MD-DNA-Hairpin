import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

from C import *

"""
Script to plot Theory-Experimental Plots of SP and PMF
"""

input_exp_x_col_name = COL_NAME_X
input_exp_sp_col_name = COL_NAME_SP
input_exp_pmf_col_name = COL_NAME_PMF

# Input ------------------------------------------
input_exp_sp_file = "data_exp/sp-2.1.csv"
input_exp_pmf_file = "data_exp/pmf-2.1.csv"
input_theory_df_file = "results-theory_exp/sp_app/sp_app-2.1.csv"
input_pmf_im_file = "results-theory_exp/sp_app/sp_app-2.1.pmf_im.csv"

bound_exp_pmf_by_pmf_im = True

interpolate_exp_sp = True
interpolate_exp_pmf = True
interpolate_exp_kind = "quadratic"
interpolate_exp_samples = 200

# Output ------------------------------------------
out_fig_file = "results-theory_exp/sp_app/sp_app-2.1.pdf"

# Main --------------------------------------------

theory_df = read_csv(input_theory_df_file)
pmf_im_df = read_csv(input_pmf_im_file)
exp_sp_df = read_csv(input_exp_sp_file)
if bound_exp_pmf_by_pmf_im:
    exp_pmf_df = load_df(input_exp_pmf_file, x_col_name=input_exp_x_col_name,
                         x_start=pmf_im_df[COL_NAME_X].values[0], x_end=pmf_im_df[COL_NAME_X].values[-1]
                         )
else:
    exp_pmf_df = read_csv(input_exp_pmf_file)

exp_sp_x = exp_sp_df[input_exp_x_col_name].values
exp_sp = exp_sp_df[input_exp_sp_col_name].values
if interpolate_exp_sp:
    exp_sp_interp_func = scipy.interpolate.interp1d(exp_sp_x, exp_sp,
                                                    kind=interpolate_exp_kind,
                                                    fill_value="extrapolate")

    exp_sp_x_interp = np.linspace(exp_sp_x[0], exp_sp_x[-1], num=interpolate_exp_samples, endpoint=True)
    exp_sp_interp = exp_sp_interp_func(exp_sp_x_interp)

exp_pmf_x = exp_pmf_df[input_exp_x_col_name].values
exp_pmf = exp_pmf_df[input_exp_pmf_col_name].values
if interpolate_exp_pmf:
    exp_pmf_interp_func = scipy.interpolate.interp1d(exp_pmf_x, exp_pmf,
                                                     kind=interpolate_exp_kind,
                                                     fill_value="extrapolate")

    exp_pmf_x_interp = np.linspace(exp_pmf_x[0], exp_pmf_x[-1], num=interpolate_exp_samples, endpoint=True)
    exp_pmf_interp = exp_pmf_interp_func(exp_pmf_x_interp)

w, h = figaspect(9 / 17)
fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
fig.tight_layout(pad=5.0)

axes[0].scatter(exp_sp_x, exp_sp,
                label="Sp (Exp by Manuel et.al)",
                # color="black"
                )
if interpolate_exp_sp:
    axes[0].plot(exp_sp_x_interp, exp_sp_interp,
                 label="Sp (Exp-Interp)", linestyle="dotted",
                 # color="black"
                 )
axes[0].plot(theory_df[COL_NAME_X],
             theory_df[COL_NAME_SP],
             label="Sp (Our Theory-Exact)")
axes[0].set_title("Splitting Probability (fold)")
axes[0].set_xlabel("x (Å)")
axes[0].set_ylabel("Sp(x)")
axes[0].legend(bbox_to_anchor=(0.2, 1.13), fontsize=7)

axes[1].scatter(exp_pmf_x, exp_pmf,
                label="PMF (Exp by Manuel et.al)",
                # color="black"
                )
if interpolate_exp_pmf:
    axes[1].plot(exp_pmf_x_interp, exp_pmf_interp,
                 label="PMF (Exp-Interp)", linestyle="dotted",
                 # color="black"
                 )
axes[1].plot(theory_df[COL_NAME_X],
             theory_df[COL_NAME_PMF_RECONSTRUCTED],
             label="PMF-Recons (Our-Theory)", linestyle="dashed")
axes[1].plot(pmf_im_df[COL_NAME_X],
             pmf_im_df[COL_NAME_PMF_IMPOSED],
             label="PMF-Imposed (Our-Theory)")
axes[1].set_title("PMF")
axes[1].set_xlabel("x (Å)")
axes[1].set_ylabel("PMF(x) (kcal/mol)")
axes[1].legend(bbox_to_anchor=(1.1, 1.15), fontsize=7)

if out_fig_file:
    plt.savefig(out_fig_file)
plt.show()
