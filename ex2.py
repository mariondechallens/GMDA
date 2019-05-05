import matplotlib.pyplot as plt
from gaussian import *
import os

savedir = f"Results/ex{2}"
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Generate data
g1 = gaussian_mix(d=2, N=3, t=1, n=500, seed=1)
g2 = gaussian_mix(d=2, N=3, t=12, n=500, seed=1)
plt.figure()
plt.scatter(g1[0], g1[1], label="g1")
plt.scatter(g2[0], g2[1], label="g2")
plt.legend()
plt.title("Ex 2 : Data generation")
plt.savefig(f"{savedir}/ex2_gaussian_data.png")

print("Ex2 done.")
