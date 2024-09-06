import matplotlib.pyplot as plt

import sp_impl
from C import *

obs_ext_pdf_file = "ext_pdf-2.2-avg50.csv"
obs_ext_col_name = COL_NAME_EXTENSION
obs_pdf_col_name = COL_NAME_PDF_AVG

p0_ext_pdf_file = "ext_pdf-2.2-avg50.csv"
p0_ext_col_name = COL_NAME_EXTENSION
p0_pdf_col_name = COL_NAME_PDF_AVG
# p0_ext_pdf_file = "sp_first_princ-fit-2.2.cond_prob_1s.csv"
# p0_ext_col_name = COL_NAME_X
# p0_pdf_col_name = COL_NAME_CONDITIONAL_PROBABILITY

out_data_file = "deconv-2.2.csv"
out_fig_file = "deconv-2.2.svg"

## Point Spread FUnction ------------------------------
kb = BOLTZMANN_CONST_KCAL_PER_MOL_K  # Boltzmann constant (kcal/mol/K) = 8.314 / (4.18 x 10-3)
temp = 300  # Temperature (K)
kb_t = kb * temp  # (kcal/mol)

ks = 10  # kcal/mol/A**2
psf_width_scale = 1
psf_stddev = psf_width_scale * ks
psf_x_mean = 0
psf_x_start = psf_x_mean + (-5 * psf_stddev)
psf_x_end = psf_x_mean + (5 * psf_stddev)

r0 = 2  # Convergence rate


def gaussian(x: np.ndarray | float, x_mean: float = 0, std_dev: float = 1) -> np.ndarray | np.float64:
    return INV_SQRT_TWO_PI * (1 / std_dev) * np.exp(-0.5 * (((x - x_mean) / std_dev) ** 2))


# Point spread function
def psf(x: np.ndarray | float) -> np.ndarray | np.float64:
    return gaussian(x=x, x_mean=psf_x_mean, std_dev=psf_stddev)


## -------------------------------------------------

def get_interp_func(x: np.ndarray, y: np.ndarray, kind: str = 'linear', fill_value=(0, 0)):
    return scipy.interpolate.interp1d(x=x, y=y, kind=kind, bounds_error=False, fill_value=fill_value)


obs_pdf_df = read_csv(obs_ext_pdf_file)
obs_ext = obs_pdf_df[obs_ext_col_name].values
obs_pdf = obs_pdf_df[obs_pdf_col_name].values
obs_interp_func = get_interp_func(obs_ext, obs_pdf)

p0_pdf_df = read_csv(p0_ext_pdf_file)
p0_ext = p0_pdf_df[p0_ext_col_name].values
p0_pdf = p0_pdf_df[p0_pdf_col_name].values
p0_interp_func = get_interp_func(p0_ext, p0_pdf)


def convolve(f, g, s: float, a: float, b: float, sample_count: int = 1000):
    def _integrand(tau):
        return f(tau) * g(s - tau)

    int_vec = np.vectorize(_integrand)

    tau_arr = np.linspace(a, b, num=sample_count, endpoint=True)
    y = int_vec(tau_arr)
    return scipy.integrate.trapezoid(y, x=tau_arr)

    # return scipy.integrate.quad(_integrand, a, b)[0]


def convolve_vec(f, g, s: np.ndarray | float,
                 a: np.ndarray | float,
                 b: np.ndarray | float,
                 sample_count: np.ndarray | int = 1000):
    _vec = np.vectorize(convolve)
    return _vec(f, g, s, a, b, sample_count)


def next_p_k(x, pk_prev_func, convolve_sample_count: int = 500):
    p_obs_x = obs_interp_func(x)
    p_prev_x = pk_prev_func(x)
    pre = r0 * (1 - (2 * abs(p_prev_x - 0.5)))

    conv = convolve(psf, pk_prev_func, s=x,
                    a=min(psf_x_start, p0_ext[0]), b=max(psf_x_end, p0_ext[-1]),
                    sample_count=convolve_sample_count)

    return p_prev_x + (pre * (p_obs_x - conv))


def iterate_pk(k_max: int, x: np.ndarray,
               pk_start_values: np.ndarray,
               k_start: int = 0,
               convolve_sample_count: int = 500,
               normalize: bool = True,
               mp_verbose: bool = False):
    pks = [pk_start_values]

    for k in range(k_start, k_max):
        prev_pk_vals = pks[k - k_start]
        prev_pk_interp_func = get_interp_func(x, y=prev_pk_vals)

        new_pk_vals = mp_execute(np.vectorize(next_p_k), x,
                                 args=(prev_pk_interp_func, convolve_sample_count),
                                 verbose=mp_verbose)
        if normalize:
            _min = np.min(new_pk_vals)
            print(f"P{k + 1} Min Value: {_min}")
            new_pk_vals -= _min

            _area = scipy.integrate.trapezoid(new_pk_vals, x)
            print(f"P{k + 1} Area: {_area}")
            if _area > 0:
                new_pk_vals /= _area

        pks.append(new_pk_vals)

    return np.array(pks)


x = p0_ext
pks = iterate_pk(2, x=x, pk_start_values=p0_pdf, k_start=0,
                 normalize=True, convolve_sample_count=500, mp_verbose=True)
out_df = pd.DataFrame()
out_df[COL_NAME_EXTENSION] = x

for k in range(0, len(pks), 10):
    out_df[COL_NAME_PDF + f"-k{k}"] = pks[k]
    # plt.plot(p0_ext, pk, label=f'P{k}')

to_csv(out_df, out_data_file)

pmfs = [sp_impl.pmf_from_pdf(pk, x=x, out_file_name=None, kb_t=kb_t) for pk in pks]
plt.plot(x, sp_impl.pmf_from_pdf(pks[0], x=x, out_file_name=None, kb_t=kb_t), label="PMF 0")
plt.plot(x, sp_impl.pmf_from_pdf(pks[-1], x=x, out_file_name=None, kb_t=kb_t), label=f"PMF {len(pks) - 1}")

plt.legend(loc='best')
plt.savefig(out_fig_file)
plt.show()
