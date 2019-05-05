import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gaussian import *
from kernel import *
from MMD import *
from feedback import *

from tqdm import tqdm
import os


def compare(g1, g2, test, param, n_permutations=20, alpha=0.95):
    """
    Parameters
    ---------
        test : MMD or JSD test
        param : kofn_power for MMD or sigma for MMD
    """
    assert test in ["JSD", "MMD"]
    if test == "JSD":
        # JSD
        rd.seed(0)
        fk = DDBC_feedback(
            g1, g2, kofn_power=param, alpha=1 - alpha
        )
        fk.permutation_test(n_permutations=n_permutations)
        return fk.b, fk.test_result, fk.p_value
    else:
        # MMD
        kernel_class = GaussianKernel(sigma=param)
        test = MMD_test(kernel_class=kernel_class, alpha_c=alpha, biased=True)

        test.fit(X=g1, Y=g2, verbose=False)

        return test.T, test.test_result


def compare_echantillon(test, param_list, N_test=50, pas=0.1, d=2, N_dim=3, N_translations=10, n=50, s=1,
                        n_permutations=20, alpha=0.95):
    assert test in ["JSD", "MMD"]

    # generate data
    data = {}
    data["g1"] = gaussian_mix(d=d, N=N_dim, t=0, n=n, s=s, seed=0)
    data["g2"] = {}
    for t in range(1, N_translations):
        data["g2"][t] = {}
        for i in range(1, N_test + 1):
            data["g2"][t][i] = gaussian_mix(d=d, N=N_dim, t=t * pas, n=n, s=s, seed=i)

    translate = np.array(range(1, N_translations)) * pas

    # for storing results
    err_type_2_df = pd.DataFrame(index=translate)
    test_stat_df = pd.DataFrame(index=translate)
    if test == "JSD":
        test_p_val_df = pd.DataFrame(index=translate)

    for t in tqdm(range(1, N_translations)):

        for param in param_list:

            test_stat_list = []
            test_result_list = []
            if test == "JSD":
                test_p_val_list = []

            for i in range(1, N_test + 1):

                comparison = compare(g1=data["g1"], g2=data["g2"][t][i], test=test, param=param,
                                     n_permutations=n_permutations, alpha=alpha)
                if test == "MMD":
                    test_stat, test_result = comparison
                else:
                    test_stat, test_result, test_p_val = comparison
                test_stat_list.append(test_stat)
                test_result_list.append(test_result)
                if test == "JSD":
                    test_p_val_list.append(test_p_val)

            test_stat_df.loc[t * pas, param] = np.mean(np.array(test_stat_list))
            err_type_2_df.loc[t * pas, param] = np.mean(np.array(test_result_list) == "accepted")
            if test == "JSD":
                test_p_val_df.loc[t * pas, param] = np.mean(np.array(test_p_val_list))

    if test == "JSD":
        return err_type_2_df, test_stat_df, test_p_val_df
    else:
        return err_type_2_df, test_stat_df


savedir = f"Results/ex{5}"
if not os.path.exists(savedir):
    os.makedirs(savedir)


###########################################################################
test = "MMD"
param_list =  [0.05, 0.1, 0.5, 1, 5, 10]
alpha=0.95
n_permutations=20
N_test=50  # number test for each configuration

# parameters for Gaussian
pas_list=[0.1, 1, 10]
d=2
N_dim=3
N_translations=10
n=50
s=1
###########################################################################
result = {}
for pas in pas_list:
    result[pas] = compare_echantillon(
        test, param_list,
        N_test=N_test, pas=pas, d=d, N_dim=N_dim, N_translations=N_translations, n=n, s=s, n_permutations=n_permutations,
        alpha=alpha
    )


err_type_2_df = pd.concat([result[pas][0] for pas in pas_list], axis=0)
test_stat_df = pd.concat([result[pas][1] for pas in pas_list], axis=0)

# display results
err_type_2_df.index = np.round(err_type_2_df.index, 5)

plt.figure()
ax = sns.heatmap(err_type_2_df, annot=True)
ax.set_title(f"Type 2 error for {test} test")
ax.set_ylabel("translation")
xlabel = "kofn power" if test == "JSD" else "Kernel sigma"
ax.set_xlabel(xlabel)
plt.savefig(f"{savedir}/ex_5_MMD_type2_error.png")
plt.close()

ax = test_stat_df.plot(title=f"Average test statistic for {test} test, for different values of kernel sigma (colors)")
ax.set_xlabel("translation")
ax.set_ylabel("Average test statistic")
ax.set_xscale('log')
plt.savefig(f"{savedir}/ex_5_MMD_average_test_statistic.png")
plt.close()

###########################################################################
test = "JSD"
param_list = [0.05, 0.1, 0.5, 0.66, 0.75, 1]
alpha=0.95
n_permutations=20
N_test=50  # number test for each configuration

# parameters for Gaussian
pas_list=[0.1, 1, 10]
d=2
N_dim=3
N_translations=10
n=50
s=1
###########################################################################
result = {}
for pas in pas_list:
    result[pas] = compare_echantillon(
        test, param_list,
        N_test=N_test, pas=pas, d=d, N_dim=N_dim, N_translations=N_translations, n=n, s=s, n_permutations=n_permutations,
        alpha=alpha
    )


err_type_2_df = pd.concat([result[pas][0] for pas in pas_list], axis=0)
test_stat_df = pd.concat([result[pas][1] for pas in pas_list], axis=0)
test_p_val_df = pd.concat([result[pas][2] for pas in pas_list], axis=0)

# display results
err_type_2_df.index = np.round(err_type_2_df.index, 5)

plt.figure()
ax = sns.heatmap(err_type_2_df, annot=True)
ax.set_title(f"Type 2 error for {test} test")
ax.set_ylabel("translation")
xlabel = "kofn power" if test == "JSD" else "Kernel sigma"
ax.set_xlabel(xlabel)
plt.savefig(f"{savedir}/ex_5_JSD_type2_error.png")
plt.close()

ax = test_stat_df.plot(title=f"Average test statistic for {test} test, for different values of kofn power (colors)")
ax.set_xlabel("translation")
ax.set_ylabel("Average test statistic")
ax.set_xscale('log')
plt.savefig(f"{savedir}/ex_5_JSD_average_test_statistic.png")
plt.close()


ax = test_p_val_df.plot(title=f"Average p-value for {test} test, for different values of kofn power (colors)")
ax.set_xlabel("translation")
ax.set_ylabel("Average p-value")
ax.set_xscale('log')
plt.savefig(f"{savedir}/ex_5_JSD_average_p_value.png")
plt.close()

print("Ex5 done.")
