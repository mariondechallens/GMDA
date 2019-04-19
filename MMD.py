import numpy as np
from kernel import Kernel


class MMD_test(object):

    """
    Maximum Mean Discrepancy test

    Inspired by "A Kernel Two-Sample Test", J. of Machine Learning Research (13), 2012, Gretton & Al.
    """

    def __init__(self, kernel_class: Kernel, alpha=0.05, biased=True):
        self.kernel_class = kernel_class
        self.alpha = alpha
        self.biased = biased
        self.T = None  # value of test statistic
        self.threshold = None  # threshold
        self.kernel = None  # kernel value
        self.X = None  # first set of data
        self.Y = None  # second set of data
        self.test_result = ""  # accepted or rejected

    def fit(self, X, Y, verbose=True):
        """
        Compute TST MMD test for a given set of samples X^(m) and Y^(n) using kernel self.kernel,
        at given level self.alpha, using self.biased or not estimator.
        :param X: first set of samples, of shape (n_features, m)
        :param Y: second set of samples, of shape (n_features, n)
        :param verbose: print results or not at the end
        """
        assert X.shape == Y.shape
        self.X = X
        self.Y = Y
        self.kernel = self.kernel_class.compute_kernel(X=self.X, Y=self.Y)

        m = X.shape[1]
        n = Y.shape[1]  # equal to m

        if self.biased:
            self.T = np.sqrt(
                np.sum(self.kernel[:m, :m]) / (m ** 2)
                + np.sum(self.kernel[m:, m:]) / (n ** 2)
                - 2 * np.sum(self.kernel[m:, :m]) / (m * n)
            )

        else:
            self.T = (
                (np.sum(self.kernel[:m, :m]) - np.sum(np.diag(self.kernel[:m, :m]))) / (m * (m - 1))
                + (np.sum(self.kernel[:n, :n]) - np.sum(np.diag(self.kernel[:n, :n]))) / (n * (n - 1))
                - 2 * np.sum(self.kernel[m:, :m]) / (m * n)
            )

        self.compute_threshold()  # threshold depends on K which depends on kernel type

        self.test_result = "accepted" if self.T <= self.threshold else "rejected"

        if verbose:
            print(self)

    def __repr__(self):
        text = f"MMD test of level alpha={self.alpha}, biased={self.biased} estimator\n"
        text += repr(self.kernel_class)
        if (self.T is not None) and (self.threshold is not None):
            if self.T <= self.threshold:
                text += f"\nResult: H0 accepted, test statistic={self.T} <= threshold={self.threshold}"
            else:
                text += f"\nResult: H0 rejected, test statistic={self.T} > threshold={self.threshold}"
        return text

    def compute_threshold(self):
        """
        :return: Acceptance threshold for H0 at threshold alpha given that the estimator is biased or not
        """
        n = self.X.shape[1]
        K = self.kernel_class.K
        if self.biased:
            self.threshold = np.sqrt(2 * K / n) * (1 + np.sqrt(2 * np.log(1 / self.alpha)))  # need n=m
        else:
            self.threshold = 4 * K * np.sqrt(np.log(1 / self.alpha) / n)

    def get_results(self):
        return {"test_result": self.test_result, "test_statistic": self.T, "threshold": self.threshold}
