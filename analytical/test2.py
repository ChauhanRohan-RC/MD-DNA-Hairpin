from C import *


def cal_mfpt(df: pd.DataFrame):
    return np.average(df[COL_NAME_TIME].values, weights=df[COL_NAME_FIRST_PASS_TIME_DISTRIBUTION].values)


df_sim = read_csv("data_sim/fpt_traj-2.csv")
df_theory = read_csv("results-theory_sim/fpt/fit-2.2.fpt_vs_t.csv")

mfpt_sim = cal_mfpt(df_sim)
mfpt_theory = cal_mfpt(df_theory)

print("## Mean First Passage Time --------------")
print(f"-> Simulation: {mfpt_sim} s")
print(f"-> Theory: {mfpt_theory} s")
print("-----------------------------------------")