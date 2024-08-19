import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.figure import figaspect

########################################################################
## Splitting Probability Sp(x) from Apparent-PMF implied by the Extension Distribution
## (by Boltzmann Inversion that assumes Equilibrium)
## Reconstruct PMF from Sp(x)
########################################################################

## Usage
# 1. Copy script to working dir
# 2. INPUT: set extension vs probability file (extension distribution)
# 3. INPUT: set Temp, LEFT and RIGHT absorbing boundaries
# 4. run with "python sp_pmf.py"
# 5. Creates output file "sp_pmf.csv"

## UNITS: Energy (kcal/mol), Distance (Å), T (K)

COMMENT_TOKEN = "#"

## Constants -------------------------
CAL_TO_JOULE = 4.184  # 1 cal = 4.184 J
K_b = 8.314 / (CAL_TO_JOULE * 1000)  # ideal gas constant in kcal/(mol K)

## INPUT -----------------------
extension_pdf_file = "ext_pdf.csv"  # Input Extension vs Probability file i.e Extension Probability
ext_col_name = "EXT"
ext_pdf_col_name = "PDF"
T = 300  # Constant Temp (K)

# First barrier (F -> I) => x_a = 14, x_b = 28
# Second barrier (I -> U) => x_a = 26, x_b = 41
x_a = 14  # TODO: LEFT Absorbing Boundary - Folded state extension (in Å)
x_b = 28  # TODO: RIGHT Absorbing Boundary - Unfolded state extension (in Å)

# Whether to negate apparent PMF for Sp(x) calculation
# -> True (default) : Sp(x) will give inflection at the potential energy barrier(s)
# -> False : Sp(x) will give inflection at the potential well(s) i.e. stable states
negate_app_pmf = True

## OUTPUT ----------------------
output_data_file = "sp_pmf1.csv"
output_fig_file = "sp_pmf1.pdf"  # (optional). Leave blank to not save figure

## ----------------------------------
# Extension Probability Distribution (PDF) DataFrame

ext_pdf_df = pd.read_csv(extension_pdf_file, sep=r"\s+", comment=COMMENT_TOKEN)

# Subset of the dataset for integration
int_df = ext_pdf_df.loc[ext_pdf_df[ext_col_name] >= x_a].loc[ext_pdf_df[ext_col_name] <= x_b]

# Apparaent PMF - PMF from extension distribution = -K_b * T * ln(P(x)) , where P(x) is extension probability distribution
pmf_apparent = np.log(int_df[ext_pdf_col_name].astype("longdouble").values) * (-K_b * T)
int_df["PMF_APP"] = pmf_apparent

# purge infinite PMF values due to PDF = 0 (or very low) values
int_df.replace([np.inf, -np.inf], np.nan, inplace=True)
int_df.dropna(subset=["PMF_APP"], axis=0, how="all", inplace=True)

# Reciprocal of PDF
if negate_app_pmf:
    pdf_recip = (1 / int_df[ext_pdf_col_name].astype(np.longdouble).values)
    int_df["PDF_RECIP"] = pdf_recip

## Calculating Spltting Probability Sp (fold)-> traj reaches folded state before unfolded state -------------------------
ext_col_index = int_df.columns.get_loc(ext_col_name)
main_col = "PDF_RECIP" if negate_app_pmf else ext_pdf_col_name  # Column which is integrated

# Integral in the denominator = Constant
c = scipy.integrate.trapezoid(y=int_df[main_col].values, x=int_df[ext_col_name].values)

sample_count = len(int_df[ext_col_name])
split_prob = np.zeros((sample_count,), dtype=np.longdouble)

for i in range(0, sample_count):
    _dist = int_df.iat[i, ext_col_index]

    _df = int_df.loc[int_df[ext_col_name] >= _dist]
    _v = scipy.integrate.trapezoid(y=_df[main_col], x=_df[ext_col_name])
    split_prob[i] = _v / c

int_df["SP"] = split_prob
# ------------------------------------------------------------------------

## Re-constructing PMF from Sp(x)
grad = np.gradient(int_df["SP"], int_df[ext_col_name])  # Gradient of Sp(fold) : ALways Negative
pmf_re = (1 if negate_app_pmf else -1) * np.log(-grad) * K_b * T
# pmf_re = (1 if negate_app_pmf else -1) * np.log(-grad * c) * K_b * T       # Accurate but does not resemble with SP traj
int_df["PMF_RE"] = pmf_re  # Reconstructed PMF from Sp(x)

# Writing Output File
if output_data_file:
    with open(output_data_file, "w") as out_p:
        out_p.write(
            f"{COMMENT_TOKEN} -------------- Splitting Probability from Simulation Apparent-PMF implied by Extension Distribution ----------------\n")
        out_p.write(f"{COMMENT_TOKEN} INPUT Extension PDF file: \"{extension_pdf_file}\"\n")
        out_p.write(f"{COMMENT_TOKEN} INPUT Absorbing Boundaries => x_a (LEFT): {x_a}  |  x_b (RIGHT): {x_b}\n")
        out_p.write(f"{COMMENT_TOKEN} INPUT Negate Apparent PMF: {negate_app_pmf}\n")
        out_p.write(f"{COMMENT_TOKEN} INPUT Thermal Energy (KbT): {K_b * T} kcal/mol/K\n")
        out_p.write(f"{COMMENT_TOKEN} ---------------------------------------\n")

        int_df[[ext_col_name,
                ext_pdf_col_name,
                "PMF_APP",
                "SP",
                "PMF_RE"]].to_csv(out_p, mode="a", sep="\t", header=True, index=False, index_label=False)

# Plotting
w, h = figaspect(12 / 16)
fig, axes = plt.subplots(2, 2, figsize=(w * 1.4, h * 1.4))
fig.tight_layout(pad=5.0)

axes[0, 0].plot(int_df[ext_col_name], int_df[ext_pdf_col_name])
axes[0, 0].set_title("Extension Distribution P(x)")
axes[0, 0].set_xlabel("Extension (Å)")
axes[0, 0].set_ylabel("P(x)")

axes[0, 1].plot(int_df[ext_col_name], int_df["PMF_APP"])
axes[0, 1].set_title("Apparent PMF = -KbT ln(P(x))")
axes[0, 1].set_xlabel("Extension (Å)")
axes[0, 1].set_ylabel("PMF (apparent)")

axes[1, 0].plot(int_df[ext_col_name], int_df["SP"])
axes[1, 0].set_title("Splitting Probability SP (fold)")
axes[1, 0].set_xlabel("Extension (Å)")
axes[1, 0].set_ylabel("Sp (fold)")

axes[1, 1].plot(int_df[ext_col_name], int_df["PMF_RE"])
axes[1, 1].set_title("Reconstructed PMF")
axes[1, 1].set_xlabel("Extension (Å)")
axes[1, 1].set_ylabel("PMF (recons)")

if output_fig_file:
    plt.savefig(output_fig_file)
plt.show()
