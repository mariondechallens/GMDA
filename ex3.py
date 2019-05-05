import matplotlib.pyplot as plt
from gaussian import *
from kernel import *
from MMD import *
from feedback import *

from tqdm import tqdm

import os

savedir = f"Results/ex{3}"
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Acceptation region shape
x = np.linspace(0.0001, 1)
y = np.sqrt(np.log(1 / x))
plt.figure()
plt.plot(x, y)
plt.title(r"Shape of $\sqrt{log(1 / \alpha_c)}$")
plt.xlabel(r'$\alpha_c$')
plt.savefig(f"{savedir}/ex3_acceptation_region_shape.png")
plt.close()

# Generate example data
g1 = gaussian_mix(d=2, N=3, t=1, n=50, seed=0)
g2 = gaussian_mix(d=2, N=3, t=1, n=50, seed=1)
plt.figure()
plt.scatter(g1[0], g1[1], label="g1")
plt.scatter(g2[0], g2[1], label="g2")
plt.legend()
plt.title("Example of data used for questions 3 and 4")
plt.savefig(f"{savedir}/ex3_data_used.png")
plt.close()

# Perform MMD test
N = 10000
for biased in [True, False]:
    for alpha_c in [0.95, 0.99]:
        res = []
        kernel_class = GaussianKernel()
        g1 = gaussian_mix(d=2, N=3, t=1, n=50, seed=0)

        for i in tqdm(range(1, N + 1)):
            g2 = gaussian_mix(d=2, N=3, t=1, n=50, seed=i)  # should be identical to g1 except for random seed

            test = MMD_test(kernel_class=kernel_class, alpha_c=alpha_c, biased=biased)
            test.fit(X=g1, Y=g2, verbose=False)

            res.append(test.test_result)

        err = [x for x in res if x == 'rejected']
        p = len(err) / N

        print(f"Results for {'MMD_b' if biased else 'MMD_u'}, alpha_c={alpha_c}")
        print(f"proba of rejecting H0 under H0={p} <= 1-alpha_c={round(1 - alpha_c, 4)} ? {p <= 1 - alpha_c}")

print("Ex3 done.")
