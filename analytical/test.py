import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

from C import *

col_name_frame = "FRAME"
col_name_ext = "EXT"

## INPUT --------------------------------------------
df1 = read_csv("data_sim/dist_vs_frame.dat", names=[col_name_frame, col_name_ext])

df2_parts = []

for i in range(1, 6):
    df2_parts.append(read_csv(f"data_sim/set2/{i}.dat", names=[col_name_frame, col_name_ext]))

# print(df2_parts)
df2 = pd.concat(df2_parts, ignore_index=True)
df2[col_name_frame] = df2.index
to_csv(df2, "data_sim/dist_vs_frame-2.dat")

# PLOT ----------------
w, h = figaspect(9 / 23)
fig, axes = plt.subplots(1, 2, figsize=(w * 1.4, h * 1.4))
fig.tight_layout(pad=5.0)

axes[0].plot(df1.index, df1.iloc[:, 1])
axes[1].plot(df2.index, df2.iloc[:, 1])

plt.show()
