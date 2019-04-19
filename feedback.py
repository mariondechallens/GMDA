import random as rd
import math
import numpy as np
from sklearn import neighbors
from collections import namedtuple


LabeledPoint = namedtuple('LabeledPoint', 'point label')


def localJSD(condProb0, theta0):
    f = lambda condP, th0: 0 if condP == 0 else math.log(condP/th0, 2)
    return condProb0 * f(condProb0, theta0) + (1-condProb0) * f(1-condProb0, 1-theta0)


def kofn_pow(n, power):
    return int(math.ceil(math.pow(n, power)))


class DDBC_feedback:

    def __init__(self, data0, data1, kofn_power=0.66, theta0=0.5, alpha=0.05, seed=0):
        """
        theta0 parameter for the random multiplexer [default: %default]', default=0.5
        kofn_power: 'Power p in  kofn=k^p, to set the number of neighbors used by the regressor [default: %default]', default=0.66
        """
        self.datasets = {0: data0.T, 1: data1.T}  # usual convention of data is (n_samples, n_features)
        self.X = np.vstack([self.datasets[0], self.datasets[1]])
        self.kofn_power = kofn_power
        self.kofn = lambda n: kofn_pow(n, self.kofn_power)
        self.theta0 = theta0
        self.labeledPoints = None
        self.locJSDTable = None
        self.local_jsd = None
        self.labels = np.hstack([np.zeros(len(self.datasets[0])), np.ones(len(self.datasets[1]))])
        self.b = None  # test statistic
        self.alpha = alpha
        self.p_value = None
        self.n_permutations = None
        self.test_result = ""
        self.jsd_obs = None
        self.jsd_permutations = None
        self.seed = seed

    def custom_feedback(self, k, labels):
        """
        Calculate JS divergence on the entire dataset, using a given list of labels. Without sampling again.
        :param k:
        :param labels:
        :return:
        """
        f = lambda cprob: localJSD(cprob, self.theta0)
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(self.X, labels)
        prob0_all = clf.predict_proba(self.X)[:, 0]  # [:, 0] is for taking the probability of predicting the 1st class
        jsd_all = list(map(f, prob0_all))  # calculate JSD for each point
        return jsd_all

    def permutation_test(self, n_permutations=10):
        """
        Permutation test without replacement as defined by B. Phipson and G.K. Smyth in "Permutation p-values should
        never be zero: calculating exact p- values when permutations are randomly drawn. ", Statistical Applications in
        Genetics and Molecular Biology, 9(1), 2010.

        :param n_permutations: number of permutations to perform
        :return:
        """
        np.random.seed(self.seed)
        self.n_permutations = n_permutations

        assert len(self.datasets[0]) == len(self.datasets[1])
        n_samples = len(self.datasets[0])
        k = self.kofn(n_samples)

        # Step 1 : estimate the JSD of the observed data
        feedback_obs = self.custom_feedback(k=k, labels=self.labels)
        jsd_obs = np.mean(feedback_obs)  # JSD of the observed points as the mean of JSD over all points

        # Step 2 : make permutations and estimate JSD for each permuation
        jsd_permutations = np.zeros(n_permutations)
        all_permutations = [list(range(2 * n_samples))]  # initialize using the labels of the observed dataset
        i = 0
        while i < self.n_permutations:
            # make permutation, only on labels
            ones_idx = rd.sample(range(2 * n_samples), n_samples)
            ones_idx = sorted(ones_idx)
            # ensure that each time we are sampling without replacement
            if ones_idx in all_permutations:
                # skip if this permutation was already used
                pass
            else:
                labels = np.zeros(2 * n_samples)
                all_permutations.append(ones_idx)
                labels[ones_idx] = 1
                # estimate JSD
                feedback_permutation = self.custom_feedback(k=k, labels=labels)
                jsd_permutations[i] = np.mean(feedback_permutation)
                i += 1

        self.jsd_obs = jsd_obs
        self.jsd_permutations = jsd_permutations

        # Step 3 : compute test statistic b : number of random datasets yielding a test statistic larger
        # than that obtained on the observed dataset.
        self.b = np.sum(jsd_permutations > jsd_obs)
        self.p_value = (self.b + 1) / (self.n_permutations + 1)
        self.test_result = "accepted" if self.p_value > self.alpha else "rejected"

    def __repr__(self):
        text = f"Feedback test of level alpha={self.alpha}\n"
        if self.test_result != "":
            text += f"Number of permutations={self.n_permutations}\n"
            text += f"jsd_obs={self.jsd_obs}, mean jsd_permutations={np.mean(self.jsd_permutations)}\n"
            if self.test_result == "accepted":
                text += f"Result: H0 accepted, test statistic={self.b}, p_value={self.p_value} >= alpha={self.alpha}"
            elif self.test_result == "rejected":
                text += f"Result: H0 rejected, test statistic={self.b}, p_value={self.p_value} < alpha={self.alpha}"
        return text

    ####################################################################################################################
    # Functions from the original implementation from SBL
    ####################################################################################################################

    def random_device(self):
        """
        Original function from SBL
        """
        raise DeprecationWarning
        counts = {0: 0, 1: 0}

        stop = False
        # generate sequence of labels until no more samples available from one population
        while not stop:
            label = rd.randint(0, 1)
            if counts[label] == self.datasets[label].shape[0]:
                stop = True
            else:
                counts[label] += 1

        # uniformly draw them from the two populations
        # create the arrays for points and labels
        nsampled = sum(counts.values())
        dim = self.datasets[0].shape[1]
        points = np.ndarray(shape=(nsampled, dim))
        labels = np.ndarray(shape=(nsampled,))

        pts_0 = rd.sample(range(self.datasets[0].shape[0]), counts[0])
        pts_1 = rd.sample(range(self.datasets[1].shape[0]), counts[1])

        points[0:counts[0], :] = self.datasets[0][pts_0]
        labels[0:counts[0]] = 0
        points[counts[0]:, :] = self.datasets[1][pts_1]
        labels[counts[0]:] = 1
        return [points, labels]

    def feedback(self, B=1):
        """
        Original function from SBL
        """
        raise DeprecationWarning
        self.B = B
        # init table where results will be stored : one row per sample point in the original pooled data set
        locJSDTable = np.ndarray(shape=(self.datasets[0].shape[0] + self.datasets[1].shape[0], self.B))

        f = lambda cprob: localJSD(cprob, self.theta0)

        # resample
        for b in range(self.B):
            # subsample datasets with random device
            pointsLabels = self.random_device()

            k = self.kofn(pointsLabels[0].shape[0])

            # init knn structure
            clf = neighbors.KNeighborsClassifier(k)

            # train knn with subsample
            clf.fit(pointsLabels[0], pointsLabels[1])

            # [:, 0] is for taking the probability of predicting the first class ie 0
            prob0_0 = clf.predict_proba(self.datasets[0])[:, 0]
            prob0_1 = clf.predict_proba(self.datasets[1])[:, 0]

            locJSDTable[0:prob0_0.shape[0], b] = list(map(f, prob0_0))
            locJSDTable[prob0_0.shape[0]:, b] = list(map(f, prob0_1))

        #         local_jsd = np.apply_along_axis(np.median, 1, locJSDTable)
        local_jsd = np.median(locJSDTable, axis=1)

        labeledPoints = []
        labeledPoints.extend(map(lambda point: LabeledPoint(point, 0), self.datasets[0]))
        labeledPoints.extend(map(lambda point: LabeledPoint(point, 1), self.datasets[1]))

        self.labeledPoints = labeledPoints
        self.locJSDTable = locJSDTable
        self.local_jsd = local_jsd