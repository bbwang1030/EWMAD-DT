import pandas as pd
from menelaus.detector import StreamingDetector
import numpy as np
# from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

class PageHinkley(StreamingDetector):
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

    def __init__(self, delta=0.01, threshold=20, burn_in=30, k=5, direction="positive"):
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
        self.k=k

        self._max = 0
        self._min = 0
        self._sum = 0
        self._mean = 0
        self.theta = np.Inf
        # self.outlier_check_before = False
        # currently, if these need to be made available, they are through the
        # to_dataframe method
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
        # print("self._mean",self._sum,self._mean,X,self.delta)
        # print("threshold,self._mean:", self.threshold,self._mean)
        self._sum = self._sum + X - self._mean - self.delta
        self.theta = self.threshold * self._mean

        # print("self._sum:", self._sum)
        # print("self._min:", self._min)
        # print("X:", X, self._mean)
        if self._sum < self._min:
            self._min = self._sum

        if self._sum > self._max:
            self._max = self._sum

        if self.direction == "positive":
            ph_difference = self._sum - self._min
            # print("ph_difference:",ph_difference)
        elif self.direction == "negative":
            ph_difference = self._max - self._sum
        # print(ph_difference,theta)
        drift_check = ph_difference > self.theta

        # if drift_check and self.samples_since_reset > self.burn_in:
        #     # print(self.samples_since_reset)
        #     self.drift_state = "drift"

        if len(self._thetas) >= self.k+1:
            diff_theta = self.theta - self._thetas[-self.k]
            if drift_check and self.samples_since_reset > self.burn_in:
                self.drift_state = "drift"
                if diff_theta > 0:
                    self.drift_state_type = "drift_begin"
                if diff_theta < 0:
                    self.drift_state_type = "incremental_end"

            self._drift_detected.append(drift_check)
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
        self._page_hinkley_differences.append(ph_difference)
        self._drift_detected.append(drift_check)
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
        self.theta = np.Inf

        self._change_scores = []
        self._page_hinkley_values = []
        self._page_hinkley_differences = []
        self._theta_threshold = []
        self._drift_detected = []
        # self._outlier_detected = []

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

class DDM_:
    def __init__(self, warning_level=2, drift_level=3, size = 10):
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.size = size
        self._change_scores = []
        self.drift_state = ""
        self.samples_since_reset = 0
        self._mean = 0
        self._thetas=[]
        self.drift_state_type=""
    def update(self, X):
        if len(self._change_scores)>=self.size+1:
            # print(self._change_scores)
            mean = np.mean(self._change_scores[(len(self._change_scores) - self.size ):-1])
            std = np.std(self._change_scores[(len(self._change_scores) - self.size ):-1])

            # 更新最小统计量
            if mean < self.p_min and mean!=0 and std!=0:
                self.p_min = mean
                self.s_min = std

            # 计算当前状态
            current_stat = mean + std
            warning_threshold = self.p_min + self.warning_level * self.s_min
            drift_threshold = self.p_min + self.drift_level * self.s_min
            # print("current_stat,drift_threshold",current_stat,drift_threshold)

            self._mean = self._mean + (X - self._mean) / self.samples_since_reset
            self.theta = self._mean
            if len(self._thetas) >= self.size + 1:
                diff_theta = self.theta - self._thetas[-self.size]

                if current_stat >= drift_threshold:
                    # print(current_stat,drift_threshold)
                    self.drift_state = "drift"
                    if diff_theta > 0:
                        self.drift_state_type = "drift_begin"
                    if diff_theta < 0:
                        self.drift_state_type = "incremental_end"

                elif current_stat >= warning_threshold :
                    self.drift_state = "warning"
                else:
                    self.drift_state = "normal"

            self._thetas.append(self.theta)
        self._change_scores.append(X)
        self.samples_since_reset = len(self._change_scores)

    def reset(self):
        """漂移后重置统计量"""
        self.p_min = float('inf')
        self.s_min = float('inf')
        self._change_scores = []
        self.drift_state = ""
        self.samples_since_reset = 0
        self._mean = 0
        self._thetas=[]
        self.drift_state_type=""



class DDM():
    """ Drift Detection Method.

    Parameters
    ----------
    min_num_instances: int (default=30)
        The minimum required number of analyzed samples so change can be
        detected. This is used to avoid false detections during the early
        moments of the detector, when the weight of one sample is important.

    warning_level: float (default=2.0)
        Warning Level

    out_control_level: float (default=3.0)
        Out-control Level

    Notes
    -----
    DDM (Drift Detection Method) [1]_ is a concept change detection method
    based on the PAC learning model premise, that the learner's error rate
    will decrease as the number of analysed samples increase, as long as the
    data distribution is stationary.

    If the algorithm detects an increase in the error rate, that surpasses
    a calculated threshold, either change is detected or the algorithm will
    warn the user that change may occur in the near future, which is called
    the warning zone.

    The detection threshold is calculated in function of two statistics,
    obtained when `(pi + si)` is minimum:

    * :math:`p_{min}`: The minimum recorded error rate.
    * `s_{min}`: The minimum recorded standard deviation.

    At instant :math:`i`, the detection algorithm uses:

    * :math:`p_i`: The error rate at instant i.
    * :math:`s_i`: The standard deviation at instant i.

    The conditions for entering the warning zone and detecting change are
    as follows:

    * if :math:`p_i + s_i \geq p_{min} + 2 * s_{min}` -> Warning zone
    * if :math:`p_i + s_i \geq p_{min} + 3 * s_{min}` -> Change detected

    References
    ----------
    .. [1] João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning
       with Drift Detection. SBIA 2004: 286-295


    """

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0):
        # super().__init__()
        self.samples_since_reset = None
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.drift_state = ""
        self._mean = 0
        self._thetas = []
        self.drift_state_type = ""
        self._score = []
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        # super().reset()
        self.samples_since_reset = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.drift_state = ""

        self._mean = 0
        self._thetas = []
        self.drift_state_type = ""

        self._score = []
    def update(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """
        # if self.in_concept_change:
        #     self.reset()
        if len(self._score) >= self.min_instances + 1:
            self.miss_prob = np.mean(self._score[(len(self._score) - self.min_instances ):-1])
            self.miss_std =np.std(self._score[(len(self._score) - self.min_instances ):-1])
            self.samples_since_reset += 1

            self.estimation = self.miss_prob
            self.delay = 0

            # if self.samples_since_reset < self.min_instances:
            #     return

            if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
                self.miss_prob_min = self.miss_prob
                self.miss_sd_min = self.miss_std
                self.miss_prob_sd_min = self.miss_prob + self.miss_std


            if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
                self.drift_state = "drift"

            elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
                self.drift_state = "warning"

            else:
                self.drift_state = "normal"

        self._score.append(prediction)

from math import *


class HDDM_A():
    """
    Drift Detection Method based on Hoeffding’s bounds with moving average-test.

    Parameters
    ----------
    drift_confidence : float (default=0.001)
        Confidence to the drift

    warning_confidence : float (default=0.005)
        Confidence to the warning

    two_side_option : bool (default=True)
        Option to monitor error increments and decrements (two-sided) or only increments (one-sided)

    Notes
    -----
    HDDM_A [1]_ is a drift detection method based on the Hoeffding’s inequality. HDDM_A uses
    the average as estimator. It receives as input a stream of real values and
    returns the estimated status of the stream: STABLE, WARNING or DRIFT.

    Implementation based on MOA [2]_.

    References
    ----------
    .. [1] Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al.
       Online and non-parametric drift detection methods based on Hoeffding’s bounds.
       IEEE Transactions on Knowledge and Data Engineering, 2014, 27(3): 810-823.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.


    """

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005,  two_side_option=True):
        # super().__init__()
        # super().reset()
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0
        self.n_estimation = 0
        self.c_estimation = 0
        self.samples_since_reset = 0

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.two_side_option = two_side_option

        self.drift_state = ""
    def update(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """
        self.total_n += 1
        self.total_c += prediction
        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c
        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c

        cota = sqrt(1.0 / (2 * self.n_min) * log(1.0 / self.drift_confidence))
        cota1 = sqrt(1.0 / (2 * self.total_n) * log(1.0 / self.drift_confidence))

        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1 :
            self.c_min = self.total_c
            self.n_min = self.total_n

        cota = sqrt(1.0 / (2 * self.n_max) * log(1.0 / self.drift_confidence))
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n

        if self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n, self.drift_confidence) :
            # print(self.total_n)
            self.n_estimation = self.total_n - self.n_min
            self.c_estimation = self.total_c - self.c_min
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            self.drift_state = "drift"
            self.in_concept_change = True
            self.in_warning_zone = False
        elif self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n, self.warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

        if self.two_side_option and self._mean_decr(self.c_max, self.n_max, self.total_c, self.total_n) :
            self.n_estimation = self.total_n - self.n_max
            self.c_estimation = self.total_c - self.c_max
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0

        self._update_estimations()
        self.samples_since_reset = self.total_n
    def _mean_incr(self, c_min, n_min, total_c, total_n, confidence):
        if n_min == total_n:
            return False

        m = (total_n - n_min) / n_min * (1.0 / total_n)
        cota = sqrt(m / 2 * log(2.0 / confidence))
        return total_c / total_n - c_min / n_min >= cota

    def _mean_decr(self, c_max, n_max, total_c, total_n):
        if n_max == total_n:
            return False

        m = (total_n - n_max) / n_max * (1.0 / total_n)
        cota = sqrt(m / 2 * log(2.0 / self.drift_confidence))
        return c_max / n_max - total_c / total_n >= cota

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        # super().reset()
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0
        self.c_estimation = 0
        self.n_estimation = 0
        self.samples_since_reset =0
        self.drift_state = ""
    def _update_estimations(self):
        """ update_estimations

        Update the length estimation and delay.

        """
        if self.total_n >= self.n_estimation:
            self.c_estimation = self.n_estimation = 0
            self.estimation = self.total_c / self.total_n
            self.delay = self.total_n
        else:
            self.estimation = self.c_estimation / self.n_estimation
            self.delay = self.n_estimation

class HDDM_W():
    """
    Drift Detection Method based on Hoeffding’s bounds with moving weighted average-test.

    Parameters
    ----------
    drift_confidence : float (default=0.001)
        Confidence to the drift

    warning_confidence : float (default=0.005)
        Confidence to the warning

    lambda_option : float (default=0.050)
        The weight given to recent data. Smaller values mean less weight given to recent data.

    two_side_option : bool (default=True)
        Option to monitor error increments and decrements (two-sided) or only increments (one-sided)

    Notes
    -----
    HDDM_W [1]_ is an online drift detection method based on McDiarmid's bounds. HDDM_W uses
    the EWMA statistic as estimator. It receives as input a stream of real predictions
    and returns the estimated status of the stream: STABLE, WARNING or DRIFT.

    Implementation based on MOA [2]_.

    References
    ----------
    .. [1] Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al.
       Online and non-parametric drift detection methods based on Hoeffding’s bounds.
       IEEE Transactions on Knowledge and Data Engineering, 2014, 27(3): 810-823.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection.hddm_w import HDDM_W
    >>> hddm_w = HDDM_W()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 1500, simulating an
    >>> # increase in error rate
    >>> for i in range(999, 1500):
    ...     data_stream[i] = 0
    >>> # Adding stream elements to HDDM_A and verifying if drift occurred
    >>> for i in range(2000):
    ...     hddm_w.add_element(data_stream[i])
    ...     if hddm_w.detected_warning_zone():
    ...         print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    ...     if hddm_w.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

    """

    class SampleInfo:
        def __init__(self):
            self.EWMA_estimator = -1.0
            self.independent_bounded_condition_sum = None

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, lambda_option=0.050, two_side_option=True):
        # super().__init__()
        # super().reset()
        self.total = self.SampleInfo()
        self.sample1_decr_monitor = self.SampleInfo()
        self.sample1_incr_monitor = self.SampleInfo()
        self.sample2_decr_monitor = self.SampleInfo()
        self.sample2_incr_monitor = self.SampleInfo()
        self.incr_cutpoint = float("inf")
        self.decr_cutpoint = float("inf")
        self.width = 0
        self.delay = 0
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_option = lambda_option
        self.two_side_option = two_side_option

        self.drift_state = ""
        self._change_scores = []
        self.samples_since_reset = 0

    def update(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Notes
        -----
        After calling self method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """
        aux_decay_rate = 1.0 - self.lambda_option
        self.width += 1
        if self.total.EWMA_estimator < 0:
            self.total.EWMA_estimator = prediction
            self.total.independent_bounded_condition_sum = 1
        else:
            self.total.EWMA_estimator = self.lambda_option * prediction + aux_decay_rate * self.total.EWMA_estimator
            self.total.independent_bounded_condition_sum = \
                self.lambda_option * self.lambda_option \
                + aux_decay_rate * aux_decay_rate * self.total.independent_bounded_condition_sum

        self._update_incr_statistics(prediction, self.drift_confidence)
        if self._monitor_mean_incr(self.drift_confidence):
            self.reset()
            self.in_concept_change = True
            self.drift_state = "drift"
            self.in_warning_zone = False
            # print(self.drift_state)
            return
        elif self._monitor_mean_incr(self.warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

        self._update_decr_statistics(prediction, self.drift_confidence)
        if self.two_side_option and self._monitor_mean_decr(self.drift_confidence):
            self.reset()
        self.estimation = self.total.EWMA_estimator

        self._change_scores.append(prediction)
        self.samples_since_reset = len(self._change_scores)


    def _detect_mean_increment(self, sample1, sample2, confidence):
        if sample1.EWMA_estimator < 0 or sample2.EWMA_estimator < 0:
            return False

        bound = sqrt((sample1.independent_bounded_condition_sum
                      + sample2.independent_bounded_condition_sum) * log(1 / confidence) / 2)
        # print(sample2.EWMA_estimator - sample1.EWMA_estimator,bound,sample2.EWMA_estimator - sample1.EWMA_estimator > bound)
        return sample2.EWMA_estimator - sample1.EWMA_estimator > bound

    def _monitor_mean_incr(self, confidence):
        return self._detect_mean_increment(self.sample1_incr_monitor, self.sample2_incr_monitor, confidence)

    def _monitor_mean_decr(self, confidence):
        return self._detect_mean_increment(self.sample2_decr_monitor, self.sample1_decr_monitor, confidence)

    def _update_incr_statistics(self, value, confidence):
        aux_decay = 1.0 - self.lambda_option
        bound = sqrt(self.total.independent_bounded_condition_sum * log(1.0 / confidence) / 2)

        if self.total.EWMA_estimator + bound < self.incr_cutpoint:
            self.incr_cutpoint = self.total.EWMA_estimator + bound
            self.sample1_incr_monitor.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_incr_monitor.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_incr_monitor = self.SampleInfo()
            self.delay = 0
        else:
            self.delay += 1
            if self.sample2_incr_monitor.EWMA_estimator < 0:
                self.sample2_incr_monitor.EWMA_estimator = value
                self.sample2_incr_monitor.independent_bounded_condition_sum = 1
            else:
                self.sample2_incr_monitor.EWMA_estimator = \
                    self.lambda_option * value + aux_decay * self.sample2_incr_monitor.EWMA_estimator
                self.sample2_incr_monitor.independent_bounded_condition_sum = \
                    self.lambda_option * self.lambda_option + \
                    aux_decay * aux_decay * self.sample2_incr_monitor.independent_bounded_condition_sum

    def _update_decr_statistics(self, value, confidence):
        aux_decay = 1.0 - self.lambda_option
        epsilon = sqrt(self.total.independent_bounded_condition_sum * log(1.0 / confidence) / 2)

        if self.total.EWMA_estimator - epsilon > self.decr_cutpoint:
            self.decr_cutpoint = self.total.EWMA_estimator - epsilon
            self.sample1_decr_monitor.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_decr_monitor.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_decr_monitor = self.SampleInfo()
        else:
            if self.sample2_decr_monitor.EWMA_estimator < 0:
                self.sample2_decr_monitor.EWMA_estimator = value
                self.sample2_decr_monitor.independent_bounded_condition_sum = 1
            else:
                self.sample2_decr_monitor.EWMA_estimator = \
                    self.lambda_option * value + aux_decay * self.sample2_decr_monitor.EWMA_estimator
                self.sample2_decr_monitor.independent_bounded_condition_sum = \
                    self.lambda_option * self.lambda_option \
                    + aux_decay * aux_decay * self.sample2_decr_monitor.independent_bounded_condition_sum

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        # super().reset()
        self.total = self.SampleInfo()
        self.sample1_decr_monitor = self.SampleInfo()
        self.sample1_incr_monitor = self.SampleInfo()
        self.sample2_decr_monitor = self.SampleInfo()
        self.sample2_incr_monitor = self.SampleInfo()
        self.incr_cutpoint = float("inf")
        self.decr_cutpoint = float("inf")
        self.width = 0
        self.delay = 0
        self.drift_state = ""
        self._change_scores = []
        self.samples_since_reset = 0

from scipy import stats

class KSWIN():
    r""" Kolmogorov-Smirnov Windowing method for concept drift detection.

    Parameters
    ----------
    alpha: float (default=0.005)
        Probability for the test statistic of the Kolmogorov-Smirnov-Test
        The alpha parameter is very sensitive, therefore should be set
        below 0.01.

    window_size: float (default=100)
        Size of the sliding window

    stat_size: float (default=30)
        Size of the statistic window

    data: numpy.ndarray of shape (n_samples, 1) (default=None,optional)
        Already collected data to avoid cold start.

    Notes
    -----
    KSWIN (Kolmogorov-Smirnov Windowing) [1]_ is a concept change detection method based
    on the Kolmogorov-Smirnov (KS) statistical test. KS-test is a statistical test with
    no assumption of underlying data distribution. KSWIN can monitor data or performance
    distributions. Note that the detector accepts one dimensional input as array.

    KSWIN maintains a sliding window :math:`\Psi` of fixed size :math:`n` (window_size). The
    last :math:`r` (stat_size) samples of :math:`\Psi` are assumed to represent the last
    concept considered as :math:`R`. From the first :math:`n-r` samples of :math:`\Psi`,
    :math:`r` samples are uniformly drawn, representing an approximated last concept :math:`W`.

    The KS-test is performed on the windows :math:`R` and :math:`W` of the same size. KS
    -test compares the distance of the empirical cumulative data distribution :math:`dist(R,W)`.

    A concept drift is detected by KSWIN if:

    * :math:`dist(R,W) > \sqrt{-\frac{ln\alpha}{r}}`

    -> The difference in empirical data distributions between the windows :math:`R` and :math:`W`
    is too large as that R and W come from the same distribution.

    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive
       Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,


    """
    def __init__(self, alpha=0.005, window_size=100, stat_size=30, data=None):
        super().__init__()
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.change_detected = False
        self.p_value = 0
        self.n = 0
        self.samples_since_reset = 0
        self.drift_state = ""
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.window_size < 0:
            raise ValueError("window_size must be greater than 0")

        if self.window_size < self.stat_size:
            raise ValueError("stat_size must be smaller than window_size")

        if type(data) != np.ndarray or type(data) is None:
            self.window = np.array([])
        else:
            self.window = data

    def update(self, input_value):
        """ Add element to sliding window

        Adds an element on top of the sliding window and removes
        the oldest one from the window. Afterwards, the KS-test
        is performed.

        Parameters
        ----------
        input_value: ndarray
            New data sample the sliding window should add.
        """
        self.n += 1
        currentLength = self.window.shape[0]
        if currentLength >= self.window_size:
            self.window = np.delete(self.window,0)
            rnd_window = np.random.choice(self.window[:-self.stat_size], self.stat_size)

            (st, self.p_value) = stats.ks_2samp(rnd_window, self.window[-self.stat_size:],mode="exact")

            if self.p_value <= self.alpha and st > 0.1:
                self.change_detected = True
                self.drift_state = "drift"
                self.window = self.window[-self.stat_size:]
                # print(self.drift_state)
            else:
                self.change_detected = False
        else: # Not enough samples in sliding window for a valid test
            self.change_detected = False

        # print(self.window,input_value.shape)
        self.window = np.concatenate([self.window,np.array([input_value]).flatten()])
        self.samples_since_reset = self.n

    def detected_change(self):
        """ Get detected change

        Returns
        -------
        bool
            Whether or not a drift occurred

        """
        return self.change_detected

    def reset(self):
        """ reset

        Resets the change detector parameters.
        """
        self.p_value = 0
        self.window = np.array([])
        self.change_detected = False
        self.drift_state = ""
        self.samples_since_reset = 0




class ADWIN():
    """ Adaptive Windowing method for concept drift detection.

    Parameters
    ----------
    delta : float (default=0.002)
        The delta parameter for the ADWIN algorithm.

    Notes
    -----
    ADWIN [1]_ (ADaptive WINdowing) is an adaptive sliding window algorithm
    for detecting change, and keeping updated statistics about a data stream.
    ADWIN allows algorithms not adapted for drifting data, to be resistant
    to this phenomenon.

    The general idea is to keep statistics from a window of variable size while
    detecting concept drift.

    The algorithm will decide the size of the window by cutting the statistics'
    window at different points and analysing the average of some statistic over
    these two windows. If the absolute value of the difference between the two
    averages surpasses a pre-defined threshold, change is detected at that point
    and all data before that time is discarded.

    References
    ----------
    .. [1] Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
       In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448.
       Society for Industrial and Applied Mathematics, 2007.



    """
    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        super().__init__()
        # default values affected by init_bucket()
        self.delta = delta
        self.last_bucket_row = 0
        self.list_row_bucket = None
        self._total = 0
        self._variance = 0
        self._width = 0
        self.bucket_number = 0

        self.__init_buckets()

        # other default values
        self.mint_min_window_longitude = 10

        self.mdbl_delta = .002
        self.mint_time = 0
        self.mdbl_width = 0

        self.detect = 0
        self._n_detections = 0
        self.detect_twice = 0
        self.mint_clock = 32

        self.bln_bucket_deleted = False
        self.bucket_num_max = 0
        self.mint_min_window_length = 5
        self.drift_state = ""
        # super().reset()
        self._change_scores=[]
        self.samples_since_reset = 0

    def reset(self):
        """ Reset detectors

        Resets statistics and adwin's window.

        Returns
        -------
        ADWIN
            self

        """
        self.__init__(delta=self.delta)
        self.drift_state = ""
        self._change_scores=[]
        self.samples_since_reset = 0
    def get_change(self):
        """ Get drift

        Returns
        -------
        bool
            Whether or not a drift occurred

        """
        return self.bln_bucket_deleted

    def reset_change(self):
        self.bln_bucket_deleted = False

    def set_clock(self, clock):
        self.mint_clock = clock

    def detected_warning_zone(self):
        return False

    @property
    def _bucket_used_bucket(self):
        return self.bucket_num_max

    @property
    def width(self):
        return self._width

    @property
    def n_detections(self):
        return self._n_detections

    @property
    def total(self):
        return self._total

    @property
    def variance(self):
        return self._variance / self._width

    @property
    def estimation(self):
        if self._width == 0:
            return 0
        return self._total / self._width

    @estimation.setter
    def estimation(self, value):
        pass

    @property
    def width_t(self):
        return self.mdbl_width

    def __init_buckets(self):
        """ Initialize the bucket's List and statistics

        Set all statistics to 0 and create a new bucket List.

        """
        self.list_row_bucket = List()
        self.last_bucket_row = 0
        self._total = 0
        self._variance = 0
        self._width = 0
        self.bucket_number = 0

        self._change_scores=[]
        self.samples_since_reset = 0
    def update(self, value):
        """ Add a new element to the sample window.

        Apart from adding the element value to the window, by inserting it in
        the correct bucket, it will also update the relevant statistics, in
        this case the total sum of all values, the window width and the total
        variance.

        Parameters
        ----------
        value: int or float (a numeric value)

        Notes
        -----
        The value parameter can be any numeric value relevant to the analysis
        of concept change. For the learners in this framework we are using
        either 0's or 1's, that are interpreted as follows:
        0: Means the learners prediction was wrong
        1: Means the learners prediction was correct

        This function should be used at every new sample analysed.

        """
        self._width += 1
        self.__insert_element_bucket(0, value, self.list_row_bucket.first)
        incremental_variance = 0

        if self._width > 1:
            incremental_variance = (self._width - 1) * (value - self._total / (self._width - 1)) * \
                                   (value - self._total / (self._width - 1)) / self._width

        self._variance += incremental_variance
        self._total += value
        self.__compress_buckets()

        """ Detects concept change in a drifting data stream.

                The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from
                Time-Changing Data with Adaptive Windowing'. The general idea is to keep
                statistics from a window of variable size while detecting concept drift.

                This function is responsible for analysing different cutting points in
                the sliding window, to verify if there is a significant change in concept.

                Returns
                -------
                bln_change : bool
                    Whether change was detected or not

                Notes
                -----
                If change was detected, one should verify the new window size, by reading
                the width property.

                """
        bln_change = False
        bln_exit = False
        bln_bucket_deleted = False
        self.mint_time += 1
        n0 = 0
        if (self.mint_time % self.mint_clock == 0) and (self.width > self.mint_min_window_longitude):
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = not bln_reduce_width
                bln_exit = False
                n0 = 0
                n1 = self._width
                u0 = 0
                u1 = self.total
                v0 = 0
                v1 = self._variance
                n2 = 0
                u2 = 0
                cursor = self.list_row_bucket.last
                i = self.last_bucket_row

                while (not bln_exit) and (cursor is not None):
                    for k in range(cursor.bucket_size_row - 1):
                        n2 = self.bucket_size(i)
                        u2 = cursor.get_total(k)

                        if n0 > 0:
                            v0 += cursor.get_variance(k) + 1. * n0 * n2 * (u0 / n0 - u2 / n2) * (u0 / n0 - u2 / n2) / (
                                    n0 + n2)

                        if n1 > 0:
                            v1 -= cursor.get_variance(k) + 1. * n1 * n2 * (u1 / n1 - u2 / n2) * (u1 / n1 - u2 / n2) / (
                                    n1 + n2)

                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if (i == 0) and (k == cursor.bucket_size_row - 1):
                            bln_exit = True
                            break

                        abs_value = 1. * ((u0 / n0) - (u1 / n1))
                        # print(abs_value,self.delta)
                        if (n1 >= self.mint_min_window_length) and (n0 >= self.mint_min_window_length) \
                                and (self.__bln_cut_expression(n0, n1, u0, u1, v0, v1, abs_value, self.delta)):
                            bln_bucket_deleted = True
                            self.detect = self.mint_time
                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time

                            bln_reduce_width = True
                            bln_change = True
                            self.drift_state = "drift"
                            if self.width > 0:
                                n0 -= self.delete_element()
                                bln_exit = True
                                break

                    cursor = cursor.get_previous()
                    i -= 1
        self.mdbl_width += self.width
        if bln_change:
            self._n_detections += 1
        self.in_concept_change = bln_change
        # return bln_change

        self._change_scores.append(value)
        self.samples_since_reset = len(self._change_scores)

    def __insert_element_bucket(self, variance, value, node):
        node.insert_bucket(value, variance)
        self.bucket_number += 1

        if self.bucket_number > self.bucket_num_max:
            self.bucket_num_max = self.bucket_number

    @staticmethod
    def bucket_size(row):
        return np.power(2, row)

    def delete_element(self):
        """ Delete an Item from the bucket list.

        Deletes the last Item and updates relevant statistics kept by ADWIN.

        Returns
        -------
        int
            The bucket size from the updated bucket

        """
        node = self.list_row_bucket.last
        n1 = self.bucket_size(self.last_bucket_row)
        self._width -= n1
        self._total -= node.get_total(0)
        u1 = node.get_total(0) / n1
        incremental_variance = node.get_variance(0) + n1 * self._width * (u1 - self._total / self._width) * \
                               (u1 - self._total / self._width) / (n1 + self._width)
        self._variance -= incremental_variance
        node.remove_bucket()
        self.bucket_number -= 1

        if node.bucket_size_row == 0:
            self.list_row_bucket.remove_from_tail()
            self.last_bucket_row -= 1

        return n1

    def __compress_buckets(self):
        cursor = self.list_row_bucket.first
        i = 0
        while cursor is not None:
            k = cursor.bucket_size_row
            if k == self.MAX_BUCKETS + 1:
                next_node = cursor.get_next_item()
                if next_node is None:
                    self.list_row_bucket.add_to_tail()
                    next_node = cursor.get_next_item()
                    self.last_bucket_row += 1
                n1 = self.bucket_size(i)
                n2 = self.bucket_size(i)
                u1 = cursor.get_total(0) / n1
                u2 = cursor.get_total(1) / n2
                incremental_variance = n1 * n2 * ((u1 - u2) * (u1 - u2)) / (n1 + n2)
                next_node.insert_bucket(cursor.get_total(0) + cursor.get_total(1), cursor.get_variance(1)
                                        + incremental_variance)
                self.bucket_number += 1
                cursor.compress_bucket_row(2)

                if next_node.bucket_size_row <= self.MAX_BUCKETS:
                    break
            else:
                break

            cursor = cursor.get_next_item()
            i += 1

    def detected_change(self):
        """ Detects concept change in a drifting data stream.

        The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from
        Time-Changing Data with Adaptive Windowing'. The general idea is to keep
        statistics from a window of variable size while detecting concept drift.

        This function is responsible for analysing different cutting points in
        the sliding window, to verify if there is a significant change in concept.

        Returns
        -------
        bln_change : bool
            Whether change was detected or not

        Notes
        -----
        If change was detected, one should verify the new window size, by reading
        the width property.

        """
        bln_change = False
        bln_exit = False
        bln_bucket_deleted = False
        self.mint_time += 1
        n0 = 0
        if (self.mint_time % self.mint_clock == 0) and (self.width > self.mint_min_window_longitude):
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = not bln_reduce_width
                bln_exit = False
                n0 = 0
                n1 = self._width
                u0 = 0
                u1 = self.total
                v0 = 0
                v1 = self._variance
                n2 = 0
                u2 = 0
                cursor = self.list_row_bucket.last
                i = self.last_bucket_row

                while (not bln_exit) and (cursor is not None):
                    for k in range(cursor.bucket_size_row - 1):
                        n2 = self.bucket_size(i)
                        u2 = cursor.get_total(k)

                        if n0 > 0:
                            v0 += cursor.get_variance(k) + 1. * n0 * n2 * (u0 / n0 - u2 / n2) * (u0 / n0 - u2 / n2) / (
                                        n0 + n2)

                        if n1 > 0:
                            v1 -= cursor.get_variance(k) + 1. * n1 * n2 * (u1 / n1 - u2 / n2) * (u1 / n1 - u2 / n2) / (
                                        n1 + n2)

                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if (i == 0) and (k == cursor.bucket_size_row - 1):
                            bln_exit = True
                            break

                        abs_value = 1. * ((u0 / n0) - (u1 / n1))
                        if (n1 >= self.mint_min_window_length) and (n0 >= self.mint_min_window_length) \
                                and (self.__bln_cut_expression(n0, n1, u0, u1, v0, v1, abs_value, self.delta)):
                            bln_bucket_deleted = True
                            self.detect = self.mint_time
                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time

                            bln_reduce_width = True
                            bln_change = True
                            print(bln_change)
                            self.drift_state = "drift"
                            if self.width > 0:
                                n0 -= self.delete_element()
                                bln_exit = True
                                break

                    cursor = cursor.get_previous()
                    i -= 1
        self.mdbl_width += self.width
        if bln_change:
            self._n_detections += 1
        self.in_concept_change = bln_change
        return bln_change

    def __bln_cut_expression(self, n0, n1, u0, u1, v0, v1, abs_value, delta):
        n = self.width
        dd = np.log(2 * np.log(n) / delta)
        v = abs(self.variance)
        m = (1. / (n0 - self.mint_min_window_length + 1)) + (1. / (n1 - self.mint_min_window_length + 1))
        epsilon = np.sqrt(2 * m * v * dd) + 1. * 2 / 3 * dd * m
        # print(np.absolute(abs_value),dd,m,v ,2 * m * v * dd,epsilon)
        return np.absolute(abs_value) > epsilon


class List(object):
    """ A linked list object for ADWIN algorithm.

    Used for storing ADWIN's bucket list. Is composed of Item objects.
    Acts as a linked list, where each element points to its predecessor
    and successor.

    """

    def __init__(self):
        super().__init__()
        self._count = None
        self._first = None
        self._last = None
        self.reset()
        self.add_to_head()

    def reset(self):
        self._count = 0
        self._first = None
        self._last = None

    def add_to_head(self):
        self._first = Item(self._first, None)
        if self._last is None:
            self._last = self._first

    def remove_from_head(self):
        self._first = self._first.get_next_item()
        if self._first is not None:
            self._first.set_previous(None)
        else:
            self._last = None
        self._count -= 1

    def add_to_tail(self):
        self._last = Item(None, self._last)
        if self._first is None:
            self._first = self._last
        self._count += 1

    def remove_from_tail(self):
        self._last = self._last.get_previous()
        if self._last is not None:
            self._last.set_next_item(None)
        else:
            self._first = None
        self._count -= 1

    @property
    def first(self):
        return self._first

    @property
    def last(self):
        return self._last

    @property
    def size(self):
        return self._count


class Item(object):
    """ Item to be used by the List object.

    The Item object, alongside the List object, are the two main data
    structures used for storing the relevant statistics for the ADWIN
    algorithm for change detection.

    Parameters
    ----------
    next_item: Item object
        Reference to the next Item in the List
    previous_item: Item object
        Reference to the previous Item in the List

    """

    def __init__(self, next_item=None, previous_item=None):
        super().__init__()
        self.next = next_item
        self.previous = previous_item
        if next_item is not None:
            next_item.previous = self
        if previous_item is not None:
            previous_item.set_next_item(self)
        self.bucket_size_row = None
        self.max_buckets = ADWIN.MAX_BUCKETS
        self.bucket_total = np.zeros(self.max_buckets + 1, dtype=float)
        self.bucket_variance = np.zeros(self.max_buckets + 1, dtype=float)
        self.reset()

    def reset(self):
        """ Reset the algorithm's statistics and window

        Returns
        -------
        ADWIN
            self

        """
        self.bucket_size_row = 0
        for i in range(ADWIN.MAX_BUCKETS + 1):
            self.__clear_buckets(i)

        return self

    def __clear_buckets(self, index):
        self.set_total(0, index)
        self.set_variance(0, index)

    def insert_bucket(self, value, variance):
        new_item = self.bucket_size_row
        self.bucket_size_row += 1
        self.set_total(value, new_item)
        self.set_variance(variance, new_item)

    def remove_bucket(self):
        self.compress_bucket_row(1)

    def compress_bucket_row(self, num_deleted=1):
        for i in range(num_deleted, ADWIN.MAX_BUCKETS + 1):
            self.bucket_total[i - num_deleted] = self.bucket_total[i]
            self.bucket_variance[i - num_deleted] = self.bucket_variance[i]

        for i in range(1, num_deleted + 1):
            self.__clear_buckets(ADWIN.MAX_BUCKETS - i + 1)

        self.bucket_size_row -= num_deleted

    def get_next_item(self):
        return self.next

    def set_next_item(self, next_item):
        self.next = next_item

    def get_previous(self):
        return self.previous

    def set_previous(self, previous):
        self.previous = previous

    def get_total(self, index):
        return self.bucket_total[index]

    def get_variance(self, index):
        return self.bucket_variance[index]

    def set_total(self, value, index):
        self.bucket_total[index] = value

    def set_variance(self, value, index):
        self.bucket_variance[index] = value
