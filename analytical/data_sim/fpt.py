"""
Script to calculate first passage time distribution from Simulation Trajectory
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

## UNITS: Distance (Å)

COMMENT_TOKEN = "#"

## INPUT -----------------------
frame_vs_ext_file = "dist_vs_frame.dat"
frame_step_fs: float = 2 * 100  # time (in fs) between frames = time_step (fs) * dcd_freq. -1 for NOT_DEFINED

# Frame Range (optional). NOTE: frame_index = time_fs / frame_step_fs
frame_index_start: int = -1  # Inclusive [-1 for no start bound]
frame_index_end: int = int(32e-9 / (frame_step_fs * 1e-15))  # Exclusive [-1 for no end bound]

fpt_frame_step: int = int(0.5e-9 / (frame_step_fs * 1e-15))  # Frames between successive FPT calculation

# Histogram parameters
x_a: float = 15  # TODO: LEFT Boundary (Å)
x_b: float = 24.23  # TODO: right Boundary (Å)
ext_bin_count: int = 1000
_ext_bin_size: float = (x_b - x_a) / ext_bin_count

normalize_pdf = False

## Output --------------------------------------------------------------
output_fpt_data_file = "fpt-1.csv"
output_fpt_fig_file = "fpt-1.pdf"

## ----------------------------------------------------------------------------

def agg_forward(arr: np.ndarray, agg_func, out_dtype=np.float32) -> np.ndarray:
    samples = len(arr) - 1
    res = np.zeros(samples, dtype=out_dtype)

    for i in range(samples):
        res[i] = agg_func(arr[i], arr[i + 1])
    return res

def agg_forward_mean(arr: np.ndarray) -> np.ndarray:
    return agg_forward(arr, lambda a,b: (a + b) / 2)


# Dataframe
frame_ext_df: pd.DataFrame = pd.read_csv(frame_vs_ext_file, sep=r"\s+", comment=COMMENT_TOKEN, names=("FRAME", "EXT"))
if frame_index_start >= 0:
    frame_ext_df = frame_ext_df[frame_ext_df["FRAME"] >= frame_index_start]

if frame_index_end >= 0:
    frame_ext_df = frame_ext_df[frame_ext_df["FRAME"] < frame_index_end]

frame_count = len(frame_ext_df["FRAME"])
print(f"Frame Count: {frame_count}")

sample_count = int(frame_count // fpt_frame_step)
print(f"FPT Sam[les: {sample_count}")

ext_series = frame_ext_df["EXT"].values

frame_arr = np.array([(i + 1) * fpt_frame_step for i in range(sample_count)])
area_arr = np.zeros((sample_count,), dtype=np.float128)

for i in range(sample_count):
    frame_end = frame_arr[i]
    _ext = ext_series[0:frame_end]

    # Histogram
    ext_hist, ext_bin_edges = np.histogram(a=_ext,
                                           bins=ext_bin_count,
                                           # range=(ext_start, ext_end),
                                           density=normalize_pdf)

    xa_bin = np.searchsorted(ext_bin_edges, x_a) - 1
    xb_bin = np.searchsorted(ext_bin_edges, x_b) - 1
    ext_vals = agg_forward_mean(ext_bin_edges)

    area = scipy.integrate.trapezoid(y=ext_hist[xa_bin: xb_bin + 1], x=ext_vals[xa_bin: xb_bin + 1])
    area_arr[i] = area

fpt_arr = -np.gradient(area_arr, frame_arr)

time_arr = None
if frame_step_fs > 0:
    time_arr = frame_arr * frame_step_fs * 1e-15

res_df = pd.DataFrame()
res_df["FRAME"] = frame_arr
if time_arr is not None:
    res_df["TIME"] = time_arr
res_df["PDF_AREA"] = area_arr
res_df["FPT"] = fpt_arr

if output_fpt_data_file:
    res_df.to_csv(output_fpt_data_file, sep="\t", header=True, index=False, index_label=False)

plt.plot(time_arr if time_arr is not None else frame_arr, fpt_arr)
plt.xlabel("$t$ (s)" if time_arr is not None else "Frame")
plt.ylabel("$P_{fpt}(t)$")
plt.suptitle("First Passage Time Distribution")
plt.title(f"using Simulation Trajectory ($x_a$: {x_a:.2f}, $x_b$: {x_b:.2f})")

if output_fpt_fig_file:
    plt.savefig(output_fpt_fig_file)
plt.show()