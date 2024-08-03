import matplotlib.axes
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os

## Constants -------------------------
CAL_TO_JOULE = 4.184  # 1 cal = 4.184 J
K_b = 8.314 / (CAL_TO_JOULE * 1000)  # ideal gas constant in kcal/(mol K)

## INPUT -----------------------
dist_vs_pdf_file = "dist_vs_pdf.dat"    # Input distance vs probability file
T = 300         # Constant Temp (K)

x_f = 0          # LEFT Absorbing Boundary - Folded state extension (in A)
x_u = 100        # RIGHT Absorbing Boundary - Unfolded state extension (in A)

## OUTPUT ----------------------
output_file = "sp_pmf.csv"

## ----------------------------------

dist_pdf_df = pd.read_csv(dist_vs_pdf_file, sep=r"\s+", comment="#", names=("DIST", "PDF"))

# SUbset of dataset for integration
int_df = dist_pdf_df.loc[dist_pdf_df["DIST"] >= x_f].loc[dist_pdf_df["DIST"] <= x_u]

# Apparaent PMF - PMF from extension distribution = -K_b * T * ln(P(x)) , where P(x) is extension probability distribution
pmf_apparent = np.log(int_df["PDF"].astype("float128").values) * (-K_b * T)
int_df["PMF_APP"] = pmf_apparent

# purge infinite PMF values due to PDF = 0 (or very low) values
int_df.replace([np.inf, -np.inf], np.nan, inplace=True)
int_df.dropna(subset=["PMF_APP"], axis=0, how="all", inplace=True)

# Reciprocal of PDF
pdf_recip = (1 / int_df["PDF"].astype(np.float128).values)
int_df["PDF_RECIP"] = pdf_recip

# Integral in the denominator = Constant
c = scipy.integrate.trapezoid(y=int_df["PDF_RECIP"].values, x=int_df["DIST"].values)
#c = scipy.integrate.trapezoid(y=int_df["PDF"].values, x=int_df["DIST"].values)

## Calculating Spltting Probability Sp(x) -------------------------
dist_col_index = int_df.columns.get_loc("DIST")
sample_count = len(int_df["DIST"])

split_prob = np.zeros((sample_count,), dtype=np.float128)

for i in range(0, sample_count):
    _dist = int_df.iat[i, dist_col_index]

    _df = int_df.loc[int_df["DIST"] >= _dist]
    _v = scipy.integrate.trapezoid(y=_df["PDF_RECIP"], x=_df["DIST"])
    # _v = scipy.integrate.trapezoid(y=_df["PDF"], x=_df["DIST"])
    split_prob[i] = _v / c

int_df["SP"] = split_prob
# ------------------------------------------------------------------------

## Re-constructing PMF from Sp(x)
grad = np.gradient(int_df["SP"], int_df["DIST"])
pmf_re = np.log(-grad) * K_b * T       # Since gradient is always negative for SP (fold)
int_df["PMF_RE"] = pmf_re              # Reconstructed PMF from Sp(x)

# Writing Output File
int_df[["DIST", "PDF", "PMF_APP", "SP", "PMF_RE"]].to_csv(output_file, sep="\t", header=True, index=False, index_label=False)

# Plotting
fig, axes  = plt.subplots(2,2)

axes[0, 0].plot(int_df["DIST"], int_df["PDF"])
axes[0, 0].set_title("Extension DIstribution")

axes[0, 1].plot(int_df["DIST"], int_df["PMF_APP"])
axes[0, 1].set_title("Apparent PMF (from extension distribution)")

axes[1, 0].plot(int_df["DIST"], int_df["SP"])
axes[1, 0].set_title("Splitting Probability")

axes[1, 1].plot(int_df["DIST"], int_df["PMF_RE"])
axes[1, 1].set_title("Reconstructed PMF")

plt.show()
