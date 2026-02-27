import numpy as np
import pandas as pd
from menelaus.detector import StreamingDetector
from scipy.stats import norm


def calculate_s(x):
    """辅助函数：仅计算 S 值"""
    x = np.asarray(x)
    # 利用广播机制计算 S，同上一段代码
    diff_matrix = x - x[:, None]
    s = np.triu(np.sign(diff_matrix), k=1).sum()
    return s


def mann_kendall_smart(x, alpha=0.1, permutation_threshold=10, num_permutations=1000):
    """
    智能 Mann-Kendall 检验
    - n >= 10: 使用正态近似 (Z-score)
    - n < 10 : 使用置换检验 (Permutation Test)
    """
    x = np.asarray(x)
    n = len(x)
    s = calculate_s(x)

    # === 分支 1: 小样本，使用置换检验 ===
    if n <= permutation_threshold:
        # print(f"[Info] Sample size {n} is small. Using Permutation Test.")

        count_extreme = 0
        for _ in range(num_permutations):
            # 随机打乱副本
            x_perm = np.random.permutation(x)
            s_perm = calculate_s(x_perm)

            # 双尾检验逻辑：统计绝对值比观测值更大(或相等)的次数
            if abs(s_perm) >= abs(s):
                count_extreme += 1

        p = count_extreme / num_permutations
        h = p < alpha
        z = 0  # 小样本下 Z 值无意义
        trend = "unknown"
        if h:
            trend = "incremental"
            # trend = "increasing" if s > 0 else "decreasing"

        return trend, h, p, z, s

    # === 分支 2: 大样本，使用正态近似 (之前的逻辑) ===
    else:
        # ... (此处填入之前提供的计算 Var(S) 和 Z 的代码) ...
        # 简写如下：
        unique_vals, counts = np.unique(x, return_counts=True)
        var_s = n * (n - 1) * (2 * n + 5) / 18
        tie_term = np.sum(counts[counts > 1] * (counts[counts > 1] - 1) * (2 * counts[counts > 1] + 5))
        var_s -= tie_term / 18

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        p = 2 * (1 - norm.cdf(abs(z)))
        h = p < alpha

        if z > 0 and h:
            trend = 'increasing'
        elif z < 0 and h:
            trend = 'decreasing'
        else:
            trend = 'no trend'

        return trend, h, p, z, s


# # === 测试 ===
# if __name__ == "__main__":
#     # 极小样本测试 (n=5)
#     tiny_data = [1, 2, 3, 5, 4]
#     # S应该很大，但是样本太小，Z-score会失效，看 Permutation 结果
#     trend, h, p, z, s = mann_kendall_smart(tiny_data)
#     print(f"Small Sample Result: Trend={trend}, S={s}, P-value={p:.4f}")

class EWMAD_DT(StreamingDetector):
    """Page-Hinkley is a univariate change detection algorithm, designed
    to detect changes in a sequential Gaussian signal. Both the running mean and
    the running Page Hinkley (PH) statistic are incremented with each
    observation. The PH stat monitors how far the current observation is from
    the running mean of all previously encountered observations, while weighting
    it by a sensitivity parameter delta. The detector alarms when the difference
    between the maximum or minimum PH statistic encountered is larger than the
    cumulative PH statistic certain threshold (xi).

    #. Increment mean with next observations
    #. Increment running sum of difference between observations and mean
    #. Compute threshold & PH statistic
    #. Enter drift or warning state if PH value is outside threshold, and the
       number of samples is greater than the burn-in requirement.

    If the threshold is too small, PH may result in many false alarms. If too
    large, the PH test will be more robust, but may miss true drift.

    Ref. :cite:t:`hinkley1971inference`
    """

    input_type = "stream"

    def __init__(self, delta=0.001, threshold=20, burn_in=30, alpha=0.1, k=5,kk=20, direction="positive"):
        """
        Args:
            delta (float, optional): Minimum amplitude of change in data needed
                to sound alarm. Defaults to 0.01.
            threshold (int, optional): Threshold for sounding alarm. Corresponds with
                PH lambda. As suggested in PCA-CD, Qahtan (2015) recommends
                setting to 1% of an appropriate window size for the dataset.
                Defaults to 20.
            burn_in (int, optional): Minimum number of data points required to
                be seen before declaring drift. Defaults to 30.
            direction (str, optional):

                * If ``'positive'``, drift is only detected for an upward change in
                  mean, when the cumulative PH statistic differs from the
                  minimum PH statistic significantly.
                * If ``'negative'``, drift is only detected for a downward change in
                  mean, when the max PH statistic differs from the cumulative
                  PH statistic significantly.

                Defaults to ``'positive'``.
        """
        super().__init__()

        self.burn_in = burn_in
        # self.delta = delta
        self.threshold = threshold
        self.direction = direction
        self.alpha = alpha
        self.theta = np.Inf
        self.delta = delta
        self.k = k
        self.kk = kk

        self._max = 0
        self._min = 0
        self._sum = 0
        self._mean = 0
        self.theta_before = np.Inf
        self._mean1 = 0
        self.ph_difference=0
        self.sum_before=0

        # currently, if these need to be made available, they are through the
        # to_dataframe method
        self.drift_state_type = ""
        self._change_scores = []
        self._thetas = []
        self._page_hinkley_values = []
        self._page_hinkley_differences = []
        self._theta_threshold = []
        self._drift_detected = []
        self._outlier_detected = []
        self._maxes = []
        self._mins = []
        self._means = []

    def update(self, X, y_true=None, y_pred=None):
        """Update the detector with a new sample.

        Args:
            X: one row of features from input data.
            y_true: one true label from input data. Not used by Page-Hinkley.
            y_pred: one predicted label from input data. Not used by Page-Hinkley.
        """
        # X_orig = X
        # X = abs(X)
        if self.drift_state == "drift":
            self.reset()
        # if self.outlier_alarm == "outlier":
        #     self._outlier_detected = []

        X, _, _ = super()._validate_input(X, None, None)
        if len(X.shape) > 1 and X.shape[1] != 1:
            raise ValueError("Page-Hinkley should only be used to monitor 1 variable.")
        super().update(X, None, None)

        self._mean = self._mean + (X - self._mean) / self.samples_since_reset
        self._mean1 = (1-self.alpha)*self._mean1 + self.alpha*(X)
        # print("threshold,self._mean:", self.threshold,self._mean)
        self._sum = (1-self.alpha)*self._sum + self.alpha* (X - self._mean)-self.delta
        # print(self.alpha*(X - self._mean) - self.delta)
        # self._sum = self._sum + X - self._mean - self.delta
        # self.theta = self.threshold * self._mean
        self.theta = self.threshold * self._mean1

        self.theta_before = self.theta
        diff_1 = self.theta - self.theta_before
        # self.i = self.i+1
        # print(self.i)
        # theta = self.threshold[0] * np.sqrt(self.alpha/(2-self.alpha))*0.001

        # print("self._sum:", self._sum)
        # print("self._min:", self._min)
        # print("X,self._mean:", X, self._mean,self._sum,self._min)
        if self._sum < self._min:
            self._min = self._sum
        # print(self._sum - self.sum_before)

        if self._sum > self._max:
            self._max = self._sum

        if self.direction == "positive":
            self.ph_difference = self._sum-self._min
            # print("ph_difference:",ph_difference)
        elif self.direction == "negative":
            self.ph_difference = self._max - self._sum

        self._page_hinkley_differences.append(self.ph_difference)
        self.sum_before = self._sum
        # print(self.ph_difference,X,self._mean,self.theta, self.ph_difference > self.theta,self.samples_since_reset)
        drift_check = self.ph_difference >  self.theta
        # print(self.ph_difference,self._sum)
        if drift_check and self.samples_since_reset > self.burn_in:
            self.drift_state = "drift"
            self.k = max(self.k, 5)
            if len(self._thetas) >= 2*self.k + 1:
                result = [x.item() for x in self._page_hinkley_differences[-self.k:]]
                trend, h, p, z, s = mann_kendall_smart(result)
                result2 = [x.item() for x in self._page_hinkley_differences[-2*self.k:]]
                diffs = np.diff(result2)
                max_gap = np.max(abs(diffs))
                max_gap_idx = np.argmax(abs(diffs))
                other_diffs = np.delete(abs(diffs), max_gap_idx)
                median_gap = np.median(other_diffs) if len(other_diffs) > 0 else 1e-9
                if median_gap == 0: median_gap = 1e-9
                ratio = max_gap / median_gap
                thres = self.kk
                if ratio > thres:
                    self.drift_state_type = "abrupt"
                elif trend == "unknown":
                    self.drift_state_type = "unknown"
                elif trend == "incremental":
                    self.drift_state_type = "incremental"
                # print(trend, result, self.ph_difference, self.theta, p,ratio,max_gap,other_diffs)
            self._drift_detected.append(drift_check)
        # print(self.ph_difference, theta ,np.sqrt(self.alpha/(2-self.alpha)), diff_theta, self.ph_difference > self.theta, self.samples_since_reset,self.drift_state_type)
        # outlier_check =  abs(X_orig-mean) > (3 * std)
        # if self.outlier_check_before and outlier_check==False and self.drift_state_before != "drift":
        #     self.outlier_alarm = "outlier"
        #     # print(self.outlier_check_before, outlier_check, self.drift_state != "drift")
        # else:
        #     self.outlier_alarm = "not_outlier"
        #     # print(self.outlier_check_before,outlier_check,self.drift_state != "drift")
        # # 在第二轮及以后，将上一轮的 outlier_check 赋给 self.outlier_check_before
        # self.outlier_check_before = outlier_check
        # self.drift_state_before = self.drift_state

        self._change_scores.append(X)
        self._thetas.append(self.theta)
        self._page_hinkley_values.append(self._sum)
        # self._page_hinkley_differences.append(self.ph_difference)
        # self._drift_detected.append(drift_check)
        self._theta_threshold.append(self.theta)
        # self._outlier_detected.append(outlier_check)

        self._maxes.append(self._max)
        self._mins.append(self._min)
        self._means.append(self._mean)

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self._max = 0
        self._min = 0
        self._sum = 0
        self._mean = 0
        self._mean1 = 0
        self.ph_difference=0
        self.sum_before=0
        self.theta=np.Inf
        self.theta_before = np.Inf

        self._change_scores = []
        self._thetas = []
        self._page_hinkley_values = []
        self._page_hinkley_differences = []
        self._theta_threshold = []
        self._drift_detected = []
        # self._outlier_detected = []
        self.drift_state_type = ""

        self._maxes = []
        self._mins = []
        self._means = []

    def to_dataframe(self):
        """Returns a dataframe storing current statistics"""
        return pd.DataFrame(
            {
                "change_scores": self._change_scores,
                "page_hinkley_values": self._page_hinkley_values,
                "page_hinkley_differences": self._page_hinkley_differences,
                "theta_threshold": self._theta_threshold,
                "drift_detected": self._drift_detected,
                "maximum_sum_values": self._maxes,
                "minimum_sum_values": self._mins,
                "mean_values": self._means,
            }
        )
