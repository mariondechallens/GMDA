from gaussian import *
from feedback import *

from tqdm import tqdm


N = 10000
n = 15

stat = []
p_val = []

g1 = gaussian_mix(d=2, N=3, t=1, n=50, seed=0)
for i in tqdm(range(1, N + 1)):
    g2 = gaussian_mix(d=2, N=3, t=1, n=50, seed=i)  # should be identical to g1 except for random seed
    rd.seed(0)
    fk = DDBC_feedback(
        g1, g2, kofn_power=0.66
    )
    fk.permutation_test(n_permutations=n)
    stat.append(fk.b)
    p_val.append(fk.p_value)

# type I error
for alpha in [0.05, 0.01]:
    print(f"Type I error for alpha={alpha}:", sum(np.array(p_val) <= alpha) / N)

print("Ex4 done.")
