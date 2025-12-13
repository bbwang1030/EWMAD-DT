import pandas as pd
from menelaus.detector import StreamingDetector
import numpy as np

class DataStream_Adapt(StreamingDetector):
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

    def __init__(self, delta=0.01, threshold=20, burn_in=30, k=10, direction="positive"):
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
        self.delta = delta
        self.threshold = threshold
        self.direction = direction
        self.k = k
        # self.outlier_alarm = "not_outlier"

        self._max = 0
        self._min = 0
        self._sum = 0
        self._mean = 0
        self.drift_state_type = ""
        # self.outlier_check_before = False
        # currently, if these need to be made available, they are through the
        # to_dataframe method
        self._change_scores = []
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
        # print("self._mean",self._sum,self._mean,X,self.delta)
        # print("threshold,self._mean:", self.threshold,self._mean)
        self._sum = max(0,self._sum + X - self._mean - self.delta)
        theta = self.threshold * self._mean

        # print("self._sum:", self._sum)
        # print("self._min:", self._min)
        # print("X:", X, self._mean)
        if self._sum < self._min:
            self._min = self._sum

        if self._sum > self._max:
            self._max = self._sum

        # if self.direction == "positive":
        #     CUSUM = max(self._sum,0)
        #     # print("ph_difference:",ph_difference)
        # elif self.direction == "negative":
        #     CUSUM = self._max - self._sum

        drift_check = self._sum > theta

        if drift_check and self.samples_since_reset > self.burn_in:
            # print(self.samples_since_reset)
            self.drift_state = "drift"

        if len(self._change_scores)>=2*self.k:
            ma_t = np.mean(self._change_scores[(len(self._change_scores) - self.k ):-1])
            ma_t_k = np.mean(self._change_scores[(len(self._change_scores) - 2*self.k ):(len(self._change_scores) - self.k)])
            if abs(ma_t - ma_t_k) > theta:
                self.drift_state_type = "incremental_end"
                self.drift_state = "drift"
        # print(CUSUM, theta / 1000,self.drift_state,self.drift_state_type,self.samples_since_reset)
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
        self._page_hinkley_values.append(self._sum)
        # self._page_hinkley_differences.append(CUSUM)
        self._drift_detected.append(drift_check)
        self._theta_threshold.append(theta)
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

        self._change_scores = []
        self._page_hinkley_values = []
        self._page_hinkley_differences = []
        self._theta_threshold = []
        self._drift_detected = []
        # self._outlier_detected = []

        self._maxes = []
        self._mins = []
        self._means = []

        self.drift_state_type = ""
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
