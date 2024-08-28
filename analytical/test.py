from C import *

df = read_csv("results-theory_sim/sp_app/sp_app-fit-1.2.sim_app_pmf_aligned.csv")
print(df[df["PMF_RE"] == df["PMF_RE"].min()])

# -------------------- sp_app-fit-1.2--------------------
## PMF_RE Theory
# maxima: 20.019416, -0.71875
# minima: 15.0, -3.604513
# barrier_energy: 2.885763

## PMF_RE Sim-Traj
# maxima: 19.729275, -0.723606
# minima: 15.133236, -3.6676
# barrier_energy: 2.943994

## PMF_RE Sim App-PMF
# maxima: 18.725, -1.861791
# minima: 15.0, -3.604494
# barrier_energy: 1.742703

# -------------------- sp_app-fit-2.2 --------------------
