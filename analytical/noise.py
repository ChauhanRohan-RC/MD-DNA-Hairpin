import pandas as pd
from matplotlib.figure import figaspect

from C import *
import numpy as np
import matplotlib.pyplot as plt

col_name_frame = "FRAME"
col_name_ext = "EXT"

## INPUT --------------------------------------------
frame_ext_df = read_csv("data_sim/dist_vs_frame.dat", names=[col_name_frame, col_name_ext])
frame_step_fs = 2 * 100  # time (in fs) between frames = time_step (fs) * dcd_freq. -1 for NOT_DEFINED

# Frame Range (optional). NOTE: frame_index = time_fs / frame_step_fs
frame_index_start = -1  # Inclusive [-1 for no start bound]
frame_index_end = -1 # 32e6 / frame_step_fs  # Exclusive [-1 for no end bound]

if frame_index_start >= 0:
    frame_ext_df = frame_ext_df[frame_ext_df[col_name_frame] >= frame_index_start]

if frame_index_end >= 0:
    frame_ext_df = frame_ext_df[frame_ext_df[col_name_frame] < frame_index_end]

# OUTPUT -------------------------------------------------------------------
out_noise_file = "data_sim/dist_vs_frame2.dat"
#----------------------------------------------

noise_add = np.random.normal(0, 1, frame_ext_df.shape[0])
print(f"NOISE MIN: {min(noise_add)}")
print(f"NOISE MAX: {max(noise_add)}")

# noise_mult = np.random.normal(1, 0.1, df.shape[0])

col_name_ext_noise = f"{col_name_ext}-NOISE"
frame_ext_df[col_name_ext_noise] = frame_ext_df[col_name_ext] + noise_add
to_csv(frame_ext_df[[col_name_frame, col_name_ext_noise]], out_noise_file)

# PLOT ----------------
w, h = figaspect(9 / 23)
fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
fig.tight_layout(pad=5.0)

axes[0].plot(frame_ext_df[col_name_frame], frame_ext_df[col_name_ext])
axes[1].plot(frame_ext_df[col_name_frame], frame_ext_df[col_name_ext_noise])

plt.show()
