from C import *
import sp_impl
import matplotlib.pyplot as plt

kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
temp = 300  # Temperature (K)
kb_t = kb * temp  # (kcal/mol)

# Input -------------
in_pmf_file_name = "data_sim/sp_traj2.2.csv"  # TODO: input pmf file
in_x_col_name = COL_NAME_EXT_BIN_MEDIAN  # COL_NAME_X, COL_NAME_EXT_BIN_MEDIAN
in_pmf_col_name = COL_NAME_PMF_RECONSTRUCTED

# Output -------------
out_pdf_file_name = "data_sim/sp_traj2.2.pdf_re.csv"  # TODO: output pdf_re file
out_x_col_name = in_x_col_name
out_pdf_col_name = COL_NAME_PDF_RECONSTRUCTED

out_fig_file_name = "data_sim/sp_traj2.2.pdf_re.pdf"

# Main ====================
df = read_csv(in_pmf_file_name)
x = df[in_x_col_name]
in_pmf = df[in_pmf_col_name]

pdf = sp_impl.pdf_from_pmf(pmf=in_pmf, x=x,
                           out_file_name=out_pdf_file_name,
                           kb_t=kb_t,
                           normalize=True, scale=1,
                           out_x_col_name=out_x_col_name,
                           out_pdf_col_name=out_pdf_col_name)

plt.plot(x, pdf, label="PDF")
plt.xlabel("x (Å)")
plt.ylabel("PDF")

if out_fig_file_name:
    plt.savefig(out_fig_file_name)
plt.show()
