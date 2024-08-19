from C import BOLTZMANN_CONST_KCAL_PER_MOL_K, COL_NAME_EXT_BIN_MEDIAN, COL_NAME_PMF_RECONSTRUCTED, COL_NAME_X, \
    COL_NAME_PMF, load_df
from double_well_pmf_fit import fit_double_well_pmf, minimize_double_well_pmf, \
    samplify_double_well_pmf_fit, load_input_pmf

"""
Script to fit the double_well_pmf model to Experimental Reconstructed-PMF

NOTE: Search TODO and set required params
"""

T = 300  # Temperature (K)
KbT = BOLTZMANN_CONST_KCAL_PER_MOL_K * T  # (kcal/mol)
Ks = 10  # Force constant (kcal/mol/Å**2)

# -> FIT Double-Well PMF
if 1:
    x, pmf, meta_str = load_input_pmf(pmf_file_path_or_buf="data_exp/pmf-2.1.csv",  # TODO: input pmf_file and col_names
                                      x_col_name=COL_NAME_X,
                                      pmf_col_name=COL_NAME_PMF,
                                      # x_col_name="x",
                                      # pmf_col_name="y",
                                      # separator=", ",
                                      x_start=None,
                                      x_end=None,
                                      sort_x=False,
                                      drop_duplicates=False,
                                      return_meta_str=True,
                                      # parsed_out_file_name="data_exp/pmf-1.1.csv",
                                      # parsed_out_df_separator="\t"
                                      )

    fit_double_well_pmf(x=x, pmf=pmf, kb_t=KbT, ks=Ks,
                        out_file_name_prefix="data_exp/pmf_fit/pmf-2.1",  # TODO: output file name prefix
                        fit_init_depth=-0.44, fit_init_bias=0.01,  # TODO: initial depth and bias
                        fit_init_x_offset=None, fit_init_x_scale=None,
                        fit_init_phi_offset=None, fit_init_phi_scale=None,
                        interpolate_pmf=True,
                        fit_interpolated_pmf=False,
                        interpolate_kind="quadratic",
                        interpolate_sample_count=200,
                        interpolate_pmf_x_extra_left=1,
                        interpolate_pmf_x_extra_right=1,
                        out_fit_samples_x_col_name=COL_NAME_X,
                        out_fit_samples_pmf_col_name=COL_NAME_PMF,
                        meta_info_str=meta_str)

## -> Find Minima of Fitted PMF
# min_val = minimize_double_well_pmf("data_exp/pmf_fit/pmf-1.1.params.txt",  # TODO: input fit-params file
#                                    kb_t=KbT, ks=Ks,
#                                    x_start=13, x_stop=16,
#                                    ret_min_value=False)
#
# print(min_val)

## -> Create Samples of Fitted PMF
# samplify_double_well_pmf_fit("data_exp/pmf_fit/pmf-1.1.params.txt",  # TODO: input fit-params file
#                              kb_t=KbT, ks=Ks,
#                              x_start=-162, x_stop=152, sample_count=200,
#                              output_sample_file="data_exp/pmf_fit/pmf-1.1.fit_samples.csv")
