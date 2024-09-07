import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect

import sp_impl
from C import *
from double_well_pmf import phi_scaled
from double_well_pmf_fit import load_fit_params

"""
Script to Evaluate quantities implemented in "sp_impl.py"

1. Calculates theoretical Splitting Probability Sp(x) and Reconstructs the-PMF using
    1. first-principles
    2. final_exact_eq
    3. apparent_pmf implied by extension distribution
    
2. Plot Theoretical and Simulation SP(x) and PMF_RECONSTRUCTED + PMF_IMPOSED
"""


class SpEval:
    def __init__(self, x_a: float, x_b: float, x_0: float, t_0: float, time_instant: float,
                 n_max: int, cyl_dn_a: float,
                 kb_t: float, ks: float,
                 beta: float,  # homogeneity coefficient beta [0, 1]
                 friction_coefficient_beta: float,
                 depth: float = -0.49, bias: float = 0,
                 x_offset: float = 0, x_scale: float = 1,
                 phi_offset: float = 0, phi_scale: float = 1,
                 x_integration_samples_first_princ: int = 100,
                 x_integration_samples_final_eq: int = 100000,
                 time_integration_start: float = 0,
                 time_integration_stop: float = 1e-4,
                 time_integration_samples: int = 200):
        """
        @param x_a: LEFT absorbing Boundary (Å)
        @param x_b: RIGHT absorbing Boundary (Å)
        @param x_0: Initial Position (Å) used in First principle calculations. Mostly = LEFT or RIGHT well
        @param t_0: Initial time (sec) used in First principle calculations
        @param time_instant: Time instant (sec) at which some time-dependent properties are calculated

        @param n_max: Max order term upto which to calculate FIRST-Principle and FINAL-EXACT equations
        @param cyl_dn_a: "a" parameter of cylindrical D_n function. See {@link sp_impl.dn()}

        @param kb_t: Thermal Energy (Kcal/mol)
        @param ks: force-constant i.e. stiffness of optical trap (kcal/mol/Å**2)
        @param beta: the homogeneity coefficient, in range [0, 1].
                    1 => completely homogenous (diffusive) medium
                    0 => completely heterogeneous (complex) medium
        @param friction_coefficient_beta: beta'th coefficient of friction [ unit: (s^beta).kcal/mol/Å**2 ], denoted by eta_beta.
                                     Optimal value In range (0.5 - 2.38) x 10-7

                                     beta'th diffusion coefficient = KbT / friction_coefficient_beta

        @param depth: "depth" parameter of the double-well pmf (unit-less). In range (-0.5, 0]
        @param bias: "bias" parameter of the double-well pmf (unit-less). Must be < critical_bias(depth)
        @param x_offset: "x_offset" of the double-well pmf (Å)
        @param x_scale: "x_scale" of the double-well pmf (unit-less)
        @param phi_offset: "phi_offset" of the double-well pmf
        @param phi_scale: "phi_scale" of the double-well pmf

        @param x_integration_samples_first_princ: no. of integration samples in x for first-principle calculations.
                                                     (expensive) Keep it low
        @param x_integration_samples_final_eq: no. of integration samples in x for final-equation calculations.
                                                Keep it high for best accuracy
        @param time_integration_start: start time of time-integrals in first-principle calculations
        @param time_integration_stop: end time of time-integrals in first-principle calculations
        @param time_integration_samples: no. of time samples for time-integrals in first-principle calculations
        """
        self.x_a = x_a  # TODO: LEFT Boundary (Å)
        self.x_b = x_b  # TODO: RIGHT Boundary (Å)

        self.x_0 = x_0  # Initially at left well
        self.t_0 = t_0  # Initial time

        self.time_instant = time_instant  # time instant to calculate conditional probability at

        # Main parameters -----------------------------
        self.kb_t = kb_t  # kcal/mol
        self.ks = ks  # Force constant (kcal/mol/Å**2)
        self.beta = beta  # Homogeneity coefficient, in range [0, 1] where 1 is fully homogenous (diffusive) media
        self.friction_coeff_beta = friction_coefficient_beta  # friction coeff (eta_beta) (unit: (s^beta) . kcal/mol/Å**2). In range (0.5 - 2.38) x 10-7
        self.n_max = n_max
        self.cyl_dn_a = cyl_dn_a  # "a" param of cylindrical function

        self.d1 = self.kb_t / self.friction_coeff_beta  # diffusion coefficient with beta=1     (in Å**2/s)
        print(f"Diffusion Coefficient D1: {self.d1} Å**2/s")

        # Fit parameters
        self.depth = depth
        self.bias = bias
        self.x_offset = x_offset
        self.x_scale = x_scale
        self.phi_offset = phi_offset
        self.phi_scale = phi_scale

        self.x_integration_samples_first_princ = x_integration_samples_first_princ
        self.x_integration_samples_final_eq = x_integration_samples_final_eq  # TODO: set integration sample count

        self.t_integration_start = time_integration_start
        self.t_integration_stop = time_integration_stop
        self.t_integration_samples = time_integration_samples

    def set_pmf_params(self, depth, bias,
                       x_offset, x_scale,
                       phi_offset, phi_scale):
        self.depth = depth
        self.bias = bias
        self.x_offset = x_offset
        self.x_scale = x_scale
        self.phi_offset = phi_offset
        self.phi_scale = phi_scale

        print("---------------------------------")
        print("SP_EVAL: PMF Parameters changed")
        _d = {'depth': depth,
              'bias': bias,
              'x_offset': x_offset,
              'x_scale': x_scale,
              'phi_offset': phi_offset,
              'phi_scale': phi_scale}
        for k, v in _d.items():
            print(f" -> {k}: {v}")
        print("---------------------------------")

    def load_pmf_fit_params(self, fit_params_file):
        self.set_pmf_params(*load_fit_params(fit_params_file))

    # Wrappers -----------------------------------------------------------------

    def phi(self, x: np.ndarray | float):
        return phi_scaled(x=x, kb_t=self.kb_t, ks=self.ks,
                          depth=self.depth, bias=self.bias,
                          x_offset=self.x_offset, x_scale=self.x_scale,
                          phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    def pmf(self, x: np.ndarray | float) -> np.ndarray | float:
        return 2 * self.kb_t * np.log(self.phi(x))

    def get_pmf_minima(self, x_start: float, x_stop: float, ret_min_value: bool = False):
        return minimize_func(self.pmf, x_start=x_start, x_stop=x_stop, ret_min_value=ret_min_value)

    def get_pmf_maxima(self, x_start: float, x_stop: float, ret_max_value: bool = False):
        return maximize_func(self.pmf, x_start=x_start, x_stop=x_stop, ret_max_value=ret_max_value)

    # Probability Density FUnction (Boltzmann Factor) from PMF
    def pdf_from_pmf(self, x: np.ndarray | float,
                     out_file_name: str | None,
                     normalize: bool = True,
                     scale: float = 1,
                     out_pdf_col_name: str = COL_NAME_PDF_RECONSTRUCTED) -> np.ndarray | float:
        pdf_arr = scale / np.square(self.phi(x))
        if normalize and isinstance(x, np.ndarray):
            s = np.sum(pdf_arr)
            if s != 0:
                pdf_arr /= s

        if out_file_name:
            _df = pd.DataFrame()
            _df[COL_NAME_X] = x
            _df[out_pdf_col_name] = pdf_arr
            to_csv(_df, out_file_name)

        return pdf_arr

    def cond_prob(self, x: np.ndarray | float, t: np.ndarray | float, normalize: bool = True,
                  x0: np.ndarray | float | None = None, t0: np.ndarray | float | None = None):

        if x0 is None:
            x0 = self.x_0

        if t0 is None:
            t0 = self.t_0

        return sp_impl.cond_prob_vec(x=x, t=t, x0=x0, t0=t0, normalize=normalize,
                                     n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                     kb_t=self.kb_t, ks=self.ks,
                                     beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                     depth=self.depth, bias=self.bias,
                                     x_offset=self.x_offset, x_scale=self.x_scale,
                                     phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    def cond_prob_integral_x(self, x0: np.ndarray | float, t0: np.ndarray | float | None = None,
                             t: np.ndarray | float | None = None):
        if t0 is None:
            t0 = self.t_0

        if t is None:
            t = self.time_instant

        return sp_impl.cond_prob_integral_x_vec(x0=x0, t0=t0, t=t,
                                                x_a=self.x_a, x_b=self.x_b,
                                                x_samples=self.x_integration_samples_first_princ,
                                                n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                                kb_t=self.kb_t, ks=self.ks,
                                                beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                                depth=self.depth, bias=self.bias,
                                                x_offset=self.x_offset, x_scale=self.x_scale,
                                                phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    def first_pass_time_first_princ(self, x0: np.ndarray | float,
                                    t0: np.ndarray | float | None = None,
                                    t: np.ndarray | float | None = None):
        if t0 is None:
            t0 = self.t_0

        if t is None:
            t = self.time_instant

        return sp_impl.first_pass_time_first_princ_vec(x0=x0, t0=t0, t=t,
                                                       x_a=self.x_a, x_b=self.x_b,
                                                       x_samples=self.x_integration_samples_first_princ,
                                                       n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                                       kb_t=self.kb_t, ks=self.ks,
                                                       beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                                       depth=self.depth, bias=self.bias,
                                                       x_offset=self.x_offset, x_scale=self.x_scale,
                                                       phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    def first_pass_time_final_eq(self, x0: np.ndarray | float | None, t: np.ndarray | float | None):
        if x0 is None:
            x0 = self.x_0

        if t is None:
            t = self.time_instant

        return sp_impl.first_pass_time_final_eq_vec(x0=x0, t=t,
                                                    x_a=self.x_a, x_b=self.x_b,
                                                    n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                                    kb_t=self.kb_t, ks=self.ks,
                                                    beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                                    depth=self.depth, bias=self.bias,
                                                    x_offset=self.x_offset, x_scale=self.x_scale,
                                                    phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    def mean_first_pass_time_final_eq(self, x0: np.ndarray | float | None):
        if x0 is None:
            x0 = self.x_0

        return sp_impl.mean_first_pass_time_final_eq_vec(x0=x0,
                                                         x_a=self.x_a, x_b=self.x_b,
                                                         n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                                         kb_t=self.kb_t, ks=self.ks,
                                                         beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                                         depth=self.depth, bias=self.bias,
                                                         x_offset=self.x_offset, x_scale=self.x_scale,
                                                         phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    # ==========================================================================================
    # ----------------------------  SPLITTING PROBABILITY Wrappers  ---------------------------
    # ==========================================================================================

    def sp_first_principle(self, out_data_file: str | None,
                           x_a: float | None = None, x_b: float | None = None,
                           x_integration_samples: int | None = None,
                           process_count: int = DEFAULT_PROCESS_COUNT,
                           return_sp_integrand: bool = True,
                           reconstruct_pmf: bool = True) -> pd.DataFrame:
        if x_a is None:
            x_a = self.x_a

        if x_b is None:
            x_b = self.x_b

        if x_integration_samples is None or x_integration_samples < 1:
            x_integration_samples = self.x_integration_samples_first_princ

        return sp_impl.sp_first_principle(x_a=x_a, x_b=x_b, x_integration_samples=x_integration_samples,
                                          t0=self.t_0, t_start=self.t_integration_start, t_stop=self.t_integration_stop,
                                          t_samples=self.t_integration_samples,
                                          process_count=process_count,
                                          return_sp_integrand=return_sp_integrand,
                                          reconstruct_pmf=reconstruct_pmf,
                                          out_data_file=out_data_file,
                                          n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                          kb_t=self.kb_t, ks=self.ks,
                                          beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                          depth=self.depth, bias=self.bias,
                                          x_offset=self.x_offset, x_scale=self.x_scale,
                                          phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    def sp_final_eq(self, out_data_file: str | None,
                    x_a: float | None = None, x_b: float | None = None,
                    x_integration_samples: int | None = None,
                    process_count: int = DEFAULT_PROCESS_COUNT,
                    return_sp_integrand: bool = True,
                    reconstruct_pmf: bool = True) -> pd.DataFrame:

        if x_a is None:
            x_a = self.x_a

        if x_b is None:
            x_b = self.x_b

        if x_integration_samples is None or x_integration_samples < 1:
            x_integration_samples = self.x_integration_samples_final_eq

        return sp_impl.sp_final_eq(x_a=x_a, x_b=x_b,
                                   x_integration_samples=x_integration_samples,
                                   process_count=process_count,
                                   return_sp_integrand=return_sp_integrand,
                                   reconstruct_pmf=reconstruct_pmf,
                                   out_data_file=out_data_file,
                                   n_max=self.n_max, cyl_dn_a=self.cyl_dn_a,
                                   kb_t=self.kb_t, ks=self.ks,
                                   beta=self.beta, friction_coeff_beta=self.friction_coeff_beta,
                                   depth=self.depth, bias=self.bias,
                                   x_offset=self.x_offset, x_scale=self.x_scale,
                                   phi_offset=self.phi_offset, phi_scale=self.phi_scale,
                                   t_integration_start=self.t_integration_start,
                                   t_integration_stop=self.t_integration_stop,
                                   t_integration_samples=self.t_integration_samples)

    def sp_apparent(self, out_data_file: str | None,
                    x_a: float | None = None, x_b: float | None = None,
                    x_integration_samples: int | None = None,
                    process_count: int = DEFAULT_PROCESS_COUNT,
                    return_sp_integrand: bool = True,
                    reconstruct_pmf: bool = True) -> pd.DataFrame:

        if x_a is None:
            x_a = self.x_a

        if x_b is None:
            x_b = self.x_b

        if x_integration_samples is None or x_integration_samples < 1:
            x_integration_samples = self.x_integration_samples_final_eq

        return sp_impl.sp_apparent(x_a=x_a, x_b=x_b,
                                   x_integration_samples=x_integration_samples,
                                   process_count=process_count,
                                   return_sp_integrand=return_sp_integrand,
                                   reconstruct_pmf=reconstruct_pmf,
                                   out_data_file=out_data_file,
                                   kb_t=self.kb_t, ks=self.ks,
                                   depth=self.depth, bias=self.bias,
                                   x_offset=self.x_offset, x_scale=self.x_scale,
                                   phi_offset=self.phi_offset, phi_scale=self.phi_scale)

    # FIRST-Principle TEST METHODS -----------------------------------------------------------

    def cal_cond_prob(self, out_data_file: str | None, out_fig_file: str | None,
                      t: float = None, normalize: bool = True,
                      x_sample_count: int = 100):
        if t is None:
            t = self.time_instant

        x = np.linspace(self.x_a, self.x_b, x_sample_count, endpoint=True)
        y = self.cond_prob(x, t=t, x0=self.x_0, t0=self.t_0, normalize=normalize)

        if out_data_file:
            df = pd.DataFrame({
                COL_NAME_X: np.round(x, 5),
                COL_NAME_CONDITIONAL_PROBABILITY: y
            })

            to_csv(df, out_data_file)

        plt.plot(x, y, label=f"t: {t} s, x0: {self.x_0} A, t0: {self.t_0} s")
        plt.xlabel("x (Å)")
        plt.ylabel("P(x, t, x0, t0)")

        plt.legend(loc="upper right")
        if out_fig_file:
            plt.savefig(out_fig_file)
        plt.show()

    def cal_cond_prob_multi_time(self, time_instants: np.ndarray,
                                 x0: float = None,
                                 normalize: bool = True,
                                 x_sample_count: int = 100,
                                 plot_title: str = "Probability Density Function",
                                 plot_subtitle: str | None = "",
                                 plot_xlabel: str = '$x$ (Å)',
                                 plot_ylabel: str = 'Probability Density $P(x)$',
                                 plot_xlims: tuple = (None, None),
                                 plot_ylims: tuple = (None, None),
                                 plot_legend_loc: str = 'upper right',
                                 time_unit_in_secs: float = 1e-6,
                                 time_unit_label: str = "µs",
                                 out_file_name_prefix: str | None = None,
                                 out_data_file: str | None = None,
                                 out_fig_file: str | None = None):

        if x0 is None:
            x0 = self.x_0

        if out_file_name_prefix:
            if not out_data_file:
                out_data_file = f"{out_file_name_prefix}.csv"
            if not out_fig_file:
                out_fig_file = f"{out_file_name_prefix}.pdf"

        pdf_arr = []

        x = np.linspace(self.x_a, self.x_b, x_sample_count, endpoint=True)
        for time in time_instants:
            pdf_arr.append(self.cond_prob(x, t=time, normalize=normalize))

        # Data frame
        if out_data_file:
            df = pd.DataFrame()
            df[COL_NAME_X] = x

            for i in range(len(time_instants)):
                df[f"{COL_NAME_CONDITIONAL_PROBABILITY}_{i}"] = pdf_arr[i]

            to_csv(df, f"{out_file_name_prefix}.csv", comments=[
                "---------------- Conditional Probability -------------------",
                f"INPUT x_a: {self.x_a} Å | x_b: {self.x_b} Å | x_0: {x0} Å",
                f"INPUT kbT: {self.kb_t}  |  Ks: {self.ks}",
                f"INPUT Double-Well PMF fit-params => depth: {self.depth} | bias: {self.bias} | x_offset: {self.x_offset} | x_scale: {self.x_scale} | phi_offset: {self.phi_offset} | phi_scale: {self.phi_scale}",
                f"OUTPUT Cond Prob Time Instants (sec): [ {', '.join(map(str, time_instants))} ]",
                f"---------------------------------------------------------"
            ])

            print(f"SP_EVAL: Conditional Probability at multiple Time instants DATA saved to file \"{out_data_file}\"")

        # Plotting
        for i in range(len(time_instants)):
            plt.plot(x, pdf_arr[i], label=f"{time_instants[i] / time_unit_in_secs:g} ${time_unit_label}$")

        plt.suptitle(plot_title)  # TODO: title and subtitle of plot
        plt.title(plot_subtitle)
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)

        if plot_xlims is not None:
            plt.gca().set_xlim(list(plot_xlims))
        if plot_ylims is not None:
            plt.gca().set_ylim(list(plot_ylims))

        plt.legend(loc=plot_legend_loc)

        if out_fig_file:
            plt.savefig(out_fig_file)
            print(f"SP_EVAL: Conditional Probability at multiple Time instants PLOT saved to file \"{out_fig_file}\"")

        plt.show()

    def _cond_prob_integral_x_vs_x0_worker(self, x0: np.ndarray | float):
        return self.cond_prob_integral_x(x0=x0, t0=self.t_0, t=self.time_instant)

    def cal_cond_prob_integral_x_vs_x0(self, x0: np.ndarray, out_data_file: str | None, out_fig_file: str | None):
        # x0 = np.linspace(self.x_a, self.x_b, 100, endpoint=True)
        y = mp_execute(self._cond_prob_integral_x_vs_x0_worker, x0, DEFAULT_PROCESS_COUNT)

        if out_data_file:
            df = pd.DataFrame({
                COL_NAME_X0: x0,
                COL_NAME_CONDITIONAL_PROBABILITY_INTEGRAL_OVER_X: y
            })

            to_csv(df, out_data_file)

        plt.plot(x0, y, label=f"t: {self.time_instant} s")
        plt.xlabel("X0")
        plt.ylabel("CP_INTx(x0, t0, t)")

        plt.legend(loc="upper right")
        if out_fig_file:
            plt.savefig(out_fig_file)
        plt.show()

    def _cond_prob_integral_x_vs_t_worker(self, t: np.ndarray | float):
        return self.cond_prob_integral_x(x0=self.x_0, t0=self.t_0, t=t)

    def cal_cond_prob_integral_x_vs_t(self, out_data_file: str | None, out_fig_file: str | None):
        t_arr = np.linspace(4.2e-8, 4e-6, 100, endpoint=False)
        y = mp_execute(self._cond_prob_integral_x_vs_t_worker, t_arr, DEFAULT_PROCESS_COUNT)

        if out_data_file:
            df = pd.DataFrame({
                COL_NAME_TIME: t_arr,
                COL_NAME_CONDITIONAL_PROBABILITY_INTEGRAL_OVER_X: y
            })

            to_csv(df, out_data_file)

        plt.plot(t_arr, y, label=f"x0: {self.x_0} A | t0: {self.t_0} s")
        plt.xlabel("t (s)")
        plt.ylabel("CP_INTx(x0, t0, t)")

        plt.legend(loc="upper right")
        if out_fig_file:
            plt.savefig(out_fig_file)
        plt.show()

    def _fpt_vs_t_first_princ_worker(self, t: np.ndarray):
        return self.first_pass_time_first_princ(x0=self.x_0, t0=self.t_0, t=t)

    def _fpt_vs_t_final_eq_worker(self, t: np.ndarray):
        return self.first_pass_time_final_eq(x0=self.x_0, t=t)

    # First passage time
    def cal_fpt_vs_t(self, t: np.ndarray, use_final_eq: bool, normalize: bool = True, out_data_file: str | None = None, out_fig_file: str | None = None):
        # NOTE: Time range for first_pass_time distribution is 40.825e-9 - 5e-6

        worker = self._fpt_vs_t_final_eq_worker if use_final_eq else self._fpt_vs_t_first_princ_worker
        fpt = mp_execute(worker, t, DEFAULT_PROCESS_COUNT)
        if normalize:
            fpt -= np.min(fpt)
            norm = np.linalg.norm(fpt)
            if norm > 0:
                fpt /= norm

        if out_data_file:
            df = pd.DataFrame({
                COL_NAME_TIME: t,
                COL_NAME_FIRST_PASS_TIME_DISTRIBUTION: fpt
            })

            to_csv(df, out_data_file,
                   comments=["---------------- First Passage Time Distribution (FPTD) -------------"])

        plt.plot(t, fpt, label=f"x0: {self.x_0}")
        plt.xlabel("Time (s)")
        plt.ylabel("First Passage Time FPT(t)")

        plt.legend(loc="upper right")
        if out_fig_file:
            plt.savefig(out_fig_file)
        plt.show()
        return fpt

    # PLOTTING -------------------------------------------------------------------------------

    def plot_pmf_imposed(self, out_data_file: str | None, out_fig_file: str | None):
        x = np.linspace(self.x_a, self.x_b, 100, endpoint=True)
        y = self.pmf(x)

        if out_data_file:
            df = pd.DataFrame({
                COL_NAME_X: x,
                COL_NAME_PMF_IMPOSED: y
            })

            to_csv(df, out_data_file)

        plt.plot(x, y, label="PMF-IM")
        plt.xlabel("x (Å)")
        plt.ylabel("PMF(x) (kcal/mol)")

        plt.legend(loc="upper right")
        if out_fig_file:
            plt.savefig(out_fig_file)
        plt.show()

    def plot_pmf_reconstructed(self, pmf_vs_x_dat_file, output_fig_file):
        df = read_csv(pmf_vs_x_dat_file)
        x = df[COL_NAME_X]
        pmf_re = df[COL_NAME_PMF_RECONSTRUCTED]
        pmf = self.pmf(x)

        plt.plot(x, pmf, label="PMF-IM")
        plt.plot(x, pmf_re, label="PMF-RE")

        plt.xlabel("x (Å)")
        plt.ylabel("PMF(x) (kcal/mol)")
        plt.legend(loc="upper right")
        if output_fig_file:
            plt.savefig(output_fig_file)
        plt.show()

    def plot_sp_theory_sim(self, sp_theory_df: pd.DataFrame,
                           sim_traj_df: pd.DataFrame | None,
                           sim_app_pmf_df: pd.DataFrame | None,
                           out_file_name_prefix: str | None,
                           out_fig_file: str | None = "",
                           plot_pmf_im: bool = True,
                           align_pmf_im: bool = True,
                           align_pmf_im_offset: float = 0,
                           pmf_im_x_extra_left: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                           pmf_im_x_extra_right: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                           sim_traj_df_col_x: str = COL_NAME_EXT_BIN_MEDIAN,
                           interp_sim_traj_sp: bool = True,
                           plot_interp_sim_traj_sp: bool = True,
                           interp_sim_traj_pmf_re: bool = True,
                           plot_interp_sim_traj_pmf_re: bool = True,
                           interp_sim_traj_samples: int = 200,
                           interp_sim_traj_x_extra_left: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                           interp_sim_traj_x_extra_right: float = 1,  # In reaction-coordinate units (mostly Angstrom)
                           sim_app_pmf_df_col_x: str = COL_NAME_EXTENSION,
                           align_sim_app_pmf: bool = True,
                           align_sim_app_pmf_left_half_only: bool = False,  # Align the minima of left-half
                           align_sim_app_pmf_right_half_only: bool = False,  # Align the minima of right-half
                           plot_sim_app_sp: bool = True,
                           plot_sim_app_pmf_re: bool = True,
                           sp_plot_title: str = "Splitting Probability (fold)",
                           pmf_plot_title: str = "PMF",
                           sp_theory_label: str = "Sp (Theory)",
                           sp_sim_traj_label: str = "Sp (Simulation-Traj)",
                           sp_sim_app_label: str = "Sp (Simulation-App_PMF)",
                           sp_sim_traj_interpolated_label: str = "Sp (Simulation-Traj-Interp)",
                           pmf_im_label: str = "PMF-Imposed (Theory)",
                           pmf_re_theory_label: str = "PMF-Recons (Theory)",
                           pmf_re_sim_traj_label: str = "PMF-Recons (Simulation-Traj)",
                           pmf_re_sim_traj_interpolated_label: str = "PMF-Recons (Simulation-Traj-Interp)",
                           pmf_re_sim_app_label: str = "PMF-Recons (Simulation-App_PMF)", ):
        """
        Plots the Splitting Probabilities (SP) and reconstructed PMF(s) from Theory and Simulation Trajectory data
        on the same plot

        In any case, PMF can be reconstructed from SP automatically if not-present, and hence not required

        -> Imposed PMF is sampled and saved to a file.
        -> Simulation SP and reconstructed PMF are interpolated and the interpolated samples are saved to files

        :param sp_theory_df: pandas DataFrame with theoretical Splitting Probability and Reconstructed PMF (optional)
                            i.e. X, SP and PMF_RE (optional) columns
                            see "sp_impl.py" methods that save this data

        :param sim_traj_df: (optional) pandas DataFrame with simulation Trajectory data: X, Splitting Probability (SP-traj)
                            and Reconstructed PMF (PMF_RE) (optional)

        :param sim_app_pmf_df: (optional) pandas DataFrame with simulation data created with Apparent-PMF approach:
                            X, Splitting Probability (SP-Apparent_PMF) and Reconstructed PMF (PMF_RE).(optional)
                            This approach assumes Equilibrium and uses Boltzmann Inversion. the workflow is...

                            -> SIm Trajectory -> PDF -> Apparent PMF -> Sp(apparent) -> Reconstructed PMF

        :param plot_pmf_im: whether to sample and plot Imposed-PMF
        :param pmf_im_x_extra_left: extra length (in Angstrom) for sampling IMPOSED-PMF to the left.
        :param pmf_im_x_extra_right: extra length (in Angstrom) for sampling IMPOSED-PMF to the right.

        :param out_file_name_prefix: (optional) prefix for output file names, like for saving newly created imposed PMF samples,
                                        interpolated simulation SP and reconstructed PMF.
                                        Set to "" or None to disable saving anything.

        :param out_fig_file: (optional) file to save the plot. If not specified, "{out_file_name_prefix}.pdf" is used

        :param sim_traj_df_col_x: column in "sim_data_df" to use as X (reaction coordinate)
        :param interp_sim_traj_sp: whether to interpolate simulation SP samples
        :param interp_sim_traj_pmf_re: whether to interpolate simulation reconstructed-PMF samples
        :param interp_sim_traj_samples: number of samples for interpolation of simulation data
        """

        x_theory = sp_theory_df[COL_NAME_X].values
        sp_theory = sp_theory_df[COL_NAME_SP].values

        if COL_NAME_PMF_RECONSTRUCTED in sp_theory_df.columns:
            pmf_re_theory = sp_theory_df[COL_NAME_PMF_RECONSTRUCTED].values
        else:
            # Reconstructing PMF automatically
            pmf_re_theory = sp_impl.pmf_re(x=x_theory, sp=sp_theory, kb_t=self.kb_t)
            sp_theory_df[COL_NAME_PMF_RECONSTRUCTED] = pmf_re_theory

        pmf_im_x = None
        pmf_im = None
        if plot_pmf_im:
            # Imposed PMF domain
            no_extra_x = pmf_im_x_extra_left == 0 and pmf_im_x_extra_right == 0
            samples_per_x = len(x_theory) / abs(x_theory[-1] - x_theory[0])

            pmf_im_x = x_theory if no_extra_x else np.concatenate((np.linspace(x_theory[0] - pmf_im_x_extra_left,
                                                                               x_theory[0],
                                                                               num=max(1, round(
                                                                                   abs(samples_per_x * pmf_im_x_extra_left))),
                                                                               endpoint=False),
                                                                   x_theory,
                                                                   np.linspace(x_theory[-1] + (1 / samples_per_x),
                                                                               x_theory[-1] + pmf_im_x_extra_right,
                                                                               num=max(1, round(
                                                                                   abs(samples_per_x * pmf_im_x_extra_right))),
                                                                               endpoint=True)))

            pmf_im = self.pmf(pmf_im_x)  # Imposed PMF
            if align_pmf_im:
                pmf_im_diff = find_overlap_y_diff(pmf_im_x, pmf_im, x_theory, pmf_re_theory)
                if pmf_im_diff is not None:
                    pmf_im += (pmf_im_diff + align_pmf_im_offset)
                    print(
                        f"SP_EVAL: Aligning Imposed-PMF with PMF_RE-theory => Actual Offset: {pmf_im_diff} | Explicit: {align_pmf_im_offset} | Total: {pmf_im_diff + align_pmf_im_offset}")

            # Save newly created Imposed-PMF samples to a file
            if out_file_name_prefix:
                df_pmf_im = pd.DataFrame({
                    COL_NAME_X: pmf_im_x,
                    COL_NAME_PMF_IMPOSED: pmf_im
                })

                _im_pmf_file_name = f"{out_file_name_prefix}.pmf_im.csv"
                to_csv(df_pmf_im, _im_pmf_file_name)
                print(f"SP_EVAL: Writing Imposed-PMF samples to file \"{_im_pmf_file_name}\"")

        x_sim_traj = None
        sp_sim_traj = None
        pmf_re_sim_traj = None
        x_sim_traj_interp = None
        sp_sim_traj_interp = None
        pmf_re_sim_traj_interp = None
        if sim_traj_df is not None:
            x_sim_traj = sim_traj_df[sim_traj_df_col_x].values
            sp_sim_traj = sim_traj_df[COL_NAME_SP].values

            if COL_NAME_PMF_RECONSTRUCTED in sim_traj_df.columns:
                pmf_re_sim_traj = sim_traj_df[COL_NAME_PMF_RECONSTRUCTED].values
            else:
                # Reconstructing PMF automatically
                pmf_re_sim_traj = sp_impl.pmf_re(x=x_sim_traj, sp=sp_sim_traj, kb_t=self.kb_t)
                sim_traj_df[COL_NAME_PMF_RECONSTRUCTED] = pmf_re_sim_traj

            if interp_sim_traj_sp or interp_sim_traj_pmf_re:
                if out_file_name_prefix:
                    df_sim_interp = pd.DataFrame()

                x_sim_traj_interp = np.linspace(x_sim_traj[0] - interp_sim_traj_x_extra_left,
                                                x_sim_traj[-1] + interp_sim_traj_x_extra_right, interp_sim_traj_samples)
                if out_file_name_prefix:
                    df_sim_interp[COL_NAME_X] = x_sim_traj_interp

                if interp_sim_traj_sp:
                    _sp_interp_func = scipy.interpolate.interp1d(x_sim_traj, sp_sim_traj, kind="quadratic",
                                                                 fill_value="extrapolate")
                    sp_sim_traj_interp = _sp_interp_func(x_sim_traj_interp)
                    if out_file_name_prefix:
                        df_sim_interp[COL_NAME_SP] = sp_sim_traj_interp

                if interp_sim_traj_pmf_re:
                    _pmf_interp_func = scipy.interpolate.interp1d(x_sim_traj, pmf_re_sim_traj, kind="quadratic",
                                                                  fill_value="extrapolate")
                    pmf_re_sim_traj_interp = _pmf_interp_func(x_sim_traj_interp)
                    if out_file_name_prefix:
                        df_sim_interp[COL_NAME_PMF_RECONSTRUCTED] = pmf_re_sim_traj_interp

                if out_file_name_prefix:
                    _sim_interp_file = f"{out_file_name_prefix}.sim_traj_interp.csv"
                    to_csv(df_sim_interp, _sim_interp_file)
                    print(
                        f"SP_EVAL: Writing interpolated Simulation-Trajectory SP and Reconstructed-PMF samples to file \"{_sim_interp_file}\"")

        x_sim_app = None
        sp_sim_app = None
        pmf_re_sim_app = None
        if sim_app_pmf_df is not None:
            x_sim_app = sim_app_pmf_df[sim_app_pmf_df_col_x].values
            sp_sim_app = sim_app_pmf_df[COL_NAME_SP].values

            if COL_NAME_PMF_RECONSTRUCTED in sim_app_pmf_df.columns:
                pmf_re_sim_app = sim_app_pmf_df[COL_NAME_PMF_RECONSTRUCTED].values
            else:
                # Reconstructing PMF automatically
                pmf_re_sim_app = sp_impl.pmf_re(x=x_sim_app, sp=sp_sim_app, kb_t=self.kb_t)
                sim_app_pmf_df[COL_NAME_PMF_RECONSTRUCTED] = pmf_re_sim_app

            if align_sim_app_pmf:
                # Align the minima of simulated_apparent_pmf_re with the minima of theory_pmf_re
                # x_ref, pmf_ref = (x_sim_traj_interp, pmf_re_sim_traj_interp) if pmf_re_sim_traj_interp is not None \
                #     else (x_theory, pmf_re_theory)

                x_ref, pmf_ref = x_theory, pmf_re_theory

                # Taking only the left-half minima of reference
                if (align_sim_app_pmf_left_half_only or align_sim_app_pmf_right_half_only) and len(x_ref) > 4:
                    ref_i_start = 0 if align_sim_app_pmf_left_half_only else len(x_ref) // 2
                    ref_i_end = len(x_ref) // 2 if align_sim_app_pmf_left_half_only else len(x_ref)
                    x_ref = x_ref[ref_i_start:ref_i_end]
                    pmf_ref = pmf_ref[ref_i_start:ref_i_end]

                _ref_min_i = np.argmin(pmf_ref)
                _ref_min_x = x_ref[_ref_min_i]
                _ref_min_pmf_re = pmf_ref[_ref_min_i]

                x_sim = x_sim_app
                pmf_sim = pmf_re_sim_app
                # Taking only the left-half minima of apparent_pmf_re
                if (align_sim_app_pmf_left_half_only or align_sim_app_pmf_right_half_only) and len(x_sim) > 4:
                    sim_i_start = 0 if align_sim_app_pmf_left_half_only else len(x_sim) // 2
                    sim_i_end = len(x_sim) // 2 if align_sim_app_pmf_left_half_only else len(x_sim)
                    x_sim = x_sim_app[sim_i_start:sim_i_end]
                    pmf_sim = pmf_re_sim_app[sim_i_start:sim_i_end]

                _sim_app_min_i = np.argmin(pmf_sim)
                _sim_app_min_x = x_sim[_sim_app_min_i]
                _sim_app_min_pmf_re = pmf_sim[_sim_app_min_i]

                x_sim_app -= (_sim_app_min_x - _ref_min_x)
                pmf_re_sim_app -= (_sim_app_min_pmf_re - _ref_min_pmf_re)

                # Create a new dataframe
                sim_app_pmf_df = sim_app_pmf_df.copy(deep=True)
                sim_app_pmf_df[sim_app_pmf_df_col_x] = x_sim_app
                sim_app_pmf_df[COL_NAME_PMF_RECONSTRUCTED] = pmf_re_sim_app

                _sim_app_pmf_file = f"{out_file_name_prefix}.sim_app_pmf_aligned.csv"
                to_csv(sim_app_pmf_df, _sim_app_pmf_file)
                print(f"SP_EVAL: writing Aligned Simulation-Apparent PMF samples to file \"{_sim_app_pmf_file}\"")

        w, h = figaspect(9 / 17)
        fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
        fig.tight_layout(pad=5.0)

        # SP Plot  -------------------------
        # axes[0].plot(x, sp_integrand, label=f"SP-INTEGRAND")
        if sim_traj_df is not None:
            axes[0].scatter(x_sim_traj, sp_sim_traj, color="black", label=sp_sim_traj_label)
            if plot_interp_sim_traj_sp and sp_sim_traj_interp is not None:
                axes[0].plot(x_sim_traj_interp, sp_sim_traj_interp, linestyle="dotted", color="black",
                             label=sp_sim_traj_interpolated_label)

        if plot_sim_app_sp and sim_app_pmf_df is not None:
            axes[0].plot(x_sim_app, sp_sim_app, label=sp_sim_app_label)

        axes[0].plot(x_theory, sp_theory, label=sp_theory_label)
        axes[0].set_title(sp_plot_title)
        axes[0].set_xlabel("x (Å)")
        axes[0].set_ylabel("Sp(x)")
        axes[0].legend(bbox_to_anchor=(0.2, 1.13), fontsize=7)

        # PMF Plot  --------------------------
        if sim_traj_df is not None:
            axes[1].scatter(x_sim_traj, pmf_re_sim_traj, color="black", label=pmf_re_sim_traj_label)
            if plot_interp_sim_traj_pmf_re and pmf_re_sim_traj_interp is not None:
                axes[1].plot(x_sim_traj_interp, pmf_re_sim_traj_interp, linestyle="dotted", color="black",
                             label=pmf_re_sim_traj_interpolated_label)

        if plot_sim_app_pmf_re and sim_app_pmf_df is not None:
            axes[1].plot(x_sim_app, pmf_re_sim_app, linestyle="solid", label=pmf_re_sim_app_label)

        axes[1].plot(x_theory, pmf_re_theory, linestyle="dashed", label=pmf_re_theory_label)
        if plot_pmf_im and not (pmf_im_x is None or pmf_im is None):
            axes[1].plot(pmf_im_x, pmf_im, label=pmf_im_label)
        axes[1].set_title(pmf_plot_title)
        axes[1].set_xlabel("x (Å)")
        axes[1].set_ylabel("PMF(x) (kcal/mol)")
        axes[1].legend(bbox_to_anchor=(1.1, 1.15), fontsize=7)

        if not out_fig_file and out_file_name_prefix:
            out_fig_file = f"{out_file_name_prefix}.pdf"

        if out_fig_file:
            plt.savefig(out_fig_file)
            print(f"SP_EVAL: SP and Reconstructed-PMF plot saved to file \"{out_fig_file}\"")

        plt.show()


# ------------------ STATIC Utility Function ------------------------

def _cal_mean_fpt(df: pd.DataFrame,
                  time_col_name: str = COL_NAME_TIME,
                  fpt_col_name: str = COL_NAME_FIRST_PASS_TIME_DISTRIBUTION):
    return np.average(df[time_col_name].values, weights=df[fpt_col_name].values)


def cal_mean_first_pass_time(theory_df_file: str,
                             sim_df_file: str | None,
                             out_file_name: str | None = "mean_fpt.txt",
                             time_col_name: str = COL_NAME_TIME,
                             fpt_col_name: str = COL_NAME_FIRST_PASS_TIME_DISTRIBUTION):
    df_theory = read_csv(theory_df_file)
    mfpt_theory = _cal_mean_fpt(df_theory,
                                time_col_name=time_col_name,
                                fpt_col_name=fpt_col_name)

    if sim_df_file:
        df_sim = read_csv(sim_df_file)
        mfpt_sim = _cal_mean_fpt(df_sim,
                                 time_col_name=time_col_name,
                                 fpt_col_name=fpt_col_name)
    else:
        df_sim = None
        mfpt_sim = None

    lines = [
        f"{COMMENT_TOKEN}{COMMENT_TOKEN} Mean First Passage Time --------------",
        f"{COMMENT_TOKEN} -> Theory: {mfpt_theory} s",
        f"{COMMENT_TOKEN} -> Simulation: {f'{mfpt_sim} s' if mfpt_sim is not None else "<No-DATA>"}",
        f"{COMMENT_TOKEN} -----------------------------------------"
    ]

    if out_file_name:
        with open(out_file_name, "w") as f:
            for line in lines:
                f.write(line + "\n")

    for line in lines:
        print(line)
