import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from detetor import PageHinkley, DDM, HDDM_A, KSWIN, ADWIN, HDDM_W, DDM_
from DataStream_Adapt import DataStream_Adapt
from EWMAD_DT import EWMAD_DT
import pandas as pd
from pyoselm.oselm import OSELMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from IPCD import IPCD
import time
from sklearn.model_selection import ParameterGrid
from sequd import SeqUD
import os
from IPCD import *
from robust_NN import *
import warnings

# 忽略所有warning
warnings.filterwarnings('ignore')
# 数据生成
np.random.seed(2)
n = 50000
bins = 50
n_i = round(n / bins)
d = 10

window_size = 500
incremental_length = 50
judge_size = 100  # 漂移后100个点为有效范围

rate = [0.01,0.02]
eta_list = [1 / 500]

# drift_type = "incremental"
# drift_type = "abrupt"
drift_type = "mixed"

epsilon = np.random.normal(0, 0.001, n).reshape(-1, 1)
np.random.seed(3)
mix_random_numbers = np.random.choice(np.arange(1, bins), size=int(bins * 0.5), replace=False)
remaining = [x for x in range(50) if x not in mix_random_numbers]

# 生成服从均匀分布的随机数
X = np.random.uniform(0.2, 0.5, (n, d))


def find_closest_values1(list_A, list_B):
    """
    O(N) 极速版：利用双指针直接匹配，无需排序。
    前提：list_A 和 list_B 必须已经是按时间从小到大排序的（通常数据流产生的数据自带此时序）。
    """
    abrupt_seq = []
    inc_starts = []
    inc_lens = []

    n_a = len(list_A)
    n_b = len(list_B)

    # B 序列的指针
    j = 0

    for i in range(n_a):
        curr_a = list_A[i]

        # 获取下一个 A 点的时间，如果是最后一个点，则设为无穷大
        next_a = list_A[i + 1] if i + 1 < n_a else float('inf')

        # 尝试匹配 B
        is_incremental = False

        # 如果 B 还有剩余，且满足: A[i] < B[j] < A[i+1]
        # 这就是 "Start 和 End 之间没有其他点" 的数学表达
        if j < n_b:
            curr_b = list_B[j]

            # 容错处理：如果 B[j] 比 A[i] 还小，说明是无效的结束点，跳过
            while j < n_b and list_B[j] <= curr_a:
                j += 1
                if j < n_b:
                    curr_b = list_B[j]

            # 再次检查边界（防止 j 越界）
            if j < n_b:
                if curr_a < curr_b < next_a:
                    is_incremental = True

        if is_incremental:
            # 匹配成功
            inc_starts.append(curr_a)
            inc_lens.append(list_B[j] - curr_a)
            j += 1  # 这个 B 用过了，指针后移
        else:
            # 匹配失败（后面没有 B，或者 B 跑到了 next_a 后面）
            abrupt_seq.append(curr_a)

    return abrupt_seq, inc_starts, inc_lens


class architecture:
    def __init__(self, model, model_name, X, Y, window_size, detector, detetor_name, eta):
        self.model = model
        self.model_name = model_name
        self.X = X
        self.Y = Y
        self.window_size = window_size
        self.detector = detector
        self.detetor_name = detetor_name
        self.eta = eta
        self.drift_points = []
        self.outlier_points = []
        self.abrupt_drift_point = []
        self.abrupt_drift_point_pre = []
        self.incremental_begin_points = []
        self.incremental_end_points = []
        self.outlier_warning = "not_outlier_warning"
        self.outlier_warning_before = False
        self.outlier_check_before = False
        self.pending_drift_idx = None

    def process(self):
        # reference_window_X = X[:self.window_size]
        # reference_window_Y = Y[:self.window_size]
        # model.fit(reference_window_X, reference_window_Y.ravel())

        res = [0]
        res_nonotlier = []
        # print(X.shape,len(X))
        for i in range(self.window_size, len(X)):
            test_window_X = self.X[i - self.window_size:i]
            test_window_Y = self.Y[i - self.window_size:i]

            if self.detector.drift_state == "drift" or i == self.window_size:
                reference_window_X = test_window_X
                reference_window_Y = test_window_Y
                if self.model_name == "$\Theta$-IPOD":
                    reference_H = np.dot(
                        np.dot(reference_window_X, inv(np.dot(reference_window_X.T, reference_window_X))),
                        reference_window_X.T
                    )
                    # print(i)
                    reference_result = IPOD_new(reference_window_X, reference_window_Y, reference_H, eta=self.eta)
                    # print(222)
                    gamma = reference_result["gamma"].reshape((self.window_size, 1))

                    self.detector.reset()

                    beta_ols = np.dot(
                        inv(np.dot(reference_window_X.T, reference_window_X)),
                        np.dot(reference_window_X.T, reference_window_Y - gamma)
                    )

                    mean = np.mean(reference_window_Y - np.dot(reference_window_X, beta_ols) - gamma)
                    std = np.std(reference_window_Y - np.dot(reference_window_X, beta_ols) - gamma)
                    # y_pred = np.dot(new_X, beta_ols)

                    # print(reference_window_X.shape,(reference_window_Y - gamma).shape,beta_ols.shape,y_pred)
                    # residual = abs((new_Y - np.dot(new_X, beta_ols))/new_Y)
                else:
                    # new_X = new_X.reshape(1, -1)
                    self.model.fit(reference_window_X, reference_window_Y.ravel())
                    mean = np.mean(reference_window_Y - self.model.predict(reference_window_X))
                    std = np.std(reference_window_Y - self.model.predict(reference_window_X))
                    self.detector.reset()
                    # y_pred = model.predict(new_X)[0]

            if self.model_name == "$\Theta$-IPOD":
                new_X = self.X[i - self.window_size]
                new_Y = self.Y[i - self.window_size]
                y_pred = np.dot(new_X, beta_ols)
                # print(abs(new_Y - y_pred),mean,std)
            else:
                new_X = self.X[i - self.window_size].reshape(1, -1)
                new_Y = self.Y[i - self.window_size]
                y_pred = self.model.predict(new_X)[0]

            residual = abs((new_Y - y_pred) / new_Y)
            # print(residual)
            res.append(float(residual))
            # detector.update(np.array([residual]))
            outlier_warning = abs(new_Y - y_pred) > mean + (2 * std)
            outlier_check = abs(new_Y - y_pred) > mean + (3 * std)

            if self.outlier_warning_before and not outlier_warning and self.drift_state_before != "drift":
                self.outlier_warning = "outlier_warning"
                if self.outlier_check_before:
                    self.outlier_warning = "outlier"
                    self.outlier_points.append(i - self.window_size - 1)
                    # print(i - self.window_size-1)
                # if (i - self.window_size-1)==58006 or (i - self.window_size-1)==58008:
                #     print(res[-2])
            # elif self.detector.drift_state == "drift":
            #     self.detector.update(np.array([res[-1]]))
            else:
                if self.detector.samples_since_reset == 0:
                    self.detector.update(np.array(0))
                    res_nonotlier.append(0)
                    # print(self.detector.samples_since_reset,np.array([res[-2]]))
                else:
                    self.outlier_warning = "not_outlier_warning"
                    self.detector.update(np.array([res[-2]]))
                    if self.detector.drift_state != "drift" and not outlier_warning:
                        res_nonotlier.append(res[-2])
                        # print(res[-2])

            self.outlier_warning_before = outlier_warning
            self.outlier_check_before = outlier_check
            self.drift_state_before = self.detector.drift_state

            distinguish = ["EWMAD_DT", "DataStream_Adapt"]
            non_distinguish = ["DDM", "PageHinkley", "HDDM_A", "HDDM_W", "KSWIN", "ADWIN"]

            if self.detetor_name == "EWMAD_DT":
                current_idx = i - self.window_size - 1
                if self.detector.drift_state_type == "incremental_end":
                    self.incremental_end_points.append(current_idx)
                    if self.pending_drift_idx is not None:
                        self.incremental_begin_points.append(self.pending_drift_idx)
                        self.pending_drift_idx = None  # 消费掉，清空
                elif self.detector.drift_state == "drift":
                    self.drift_points.append(current_idx)
                    self.pending_drift_idx = current_idx
                    self.abrupt_drift_point_pre.append(self.pending_drift_idx)
                self.abrupt_drift_point = list(set(self.abrupt_drift_point_pre) - set(self.incremental_begin_points))
            elif self.detetor_name == "DataStream_Adapt":
                if self.detector.drift_state == "drift" and self.detector.drift_state_type != "incremental_end":
                    self.abrupt_drift_point.append(i - self.window_size - 1)
                elif self.detector.drift_state_type == "incremental_end":
                    # print(self.incremental_begin_points)
                    self.incremental_begin_points.append(i - self.window_size - 1)
                # self.abrupt_drift_point = self.drift_points
                # self.incremental_begin_points = self.incremental_end_points
                self.drift_points = self.abrupt_drift_point + self.incremental_begin_points
                self.incremental_end_points = []
            elif self.detetor_name in non_distinguish:
                # print(self.detetor_name)
                if self.detector.drift_state == "drift":
                    # print(i - self.window_size - 1)
                    self.drift_points.append(i - self.window_size - 1)
                self.incremental_end_points = []
                drift_points = self.drift_points
                self.abrupt_drift_point = []
                self.incremental_begin_points = []
                self.incremental_end_points = []

            # if outlier_warning != True:
            #     res_nonotlier.append(residual)

        return self.drift_points, self.abrupt_drift_point, self.incremental_begin_points, self.incremental_end_points, self.outlier_points, res_nonotlier, res


def F1_score(drift_points, true_drift_points, judge_size):
    """
    计算 F1 Score，规则：
    1. 在真实漂移点后的 judge_size 范围内，第一个检测到的点算 TP。
    2. 同一范围内后续的检测点算 FP (重复报警)。
    3. 不在范围内的检测点算 FP。
    4. 没有被检测到的真实漂移点算 FN。
    """
    TP = 0
    FP = 0
    FN = 0
    delay = []

    # 用于记录哪些真实漂移点已经被"认领"了
    detected_true_drifts = set()

    # 为了确保逻辑顺序，建议先排序（如果输入已经是排序的可忽略）
    drift_points = sorted(drift_points)
    true_drift_points = sorted(true_drift_points)

    for detected in drift_points:
        is_match = False  # 标记当前检测点是否匹配到了某个真实漂移

        for actual in true_drift_points:
            # 判断是否在有效窗口内：(检测点 - 真实点) 介于 0 到 judge_size 之间
            if 0 <= (detected - actual) <= judge_size:
                is_match = True

                if actual not in detected_true_drifts:
                    # Case 1: 这是一个新的、合法的检测 -> TP
                    TP += 1
                    detected_true_drifts.add(actual)  # 标记该真实点已被检测
                    delay.append(detected - actual)
                else:
                    # Case 2: 这个真实点之前已经被检测过了，这是重复报警 -> FP
                    FP += 1

                # 既然已经匹配到了这个 detected 对应的 actual，就跳出内层循环
                # 防止一个检测点同时对应两个非常接近的真实漂移点（虽然很少见）
                break

        if not is_match:
            # Case 3: 该检测点不在任何真实漂移的窗口内 -> FP
            FP += 1

    # 计算 FN: 总真实点数量 - 被成功检测到的真实点数量
    FN = len(true_drift_points) - len(detected_true_drifts)

    # 计算 Precision, Recall, F1
    if TP != 0:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        F1_Score = 0
        Precision = 0
        Recall = 0

    return F1_Score, Precision, Recall, TP, FP, FN, delay


def set_params(name_detector, params):
    if name_detector == "ADWIN":
        detector = ADWIN(**params)
    elif name_detector == "DDM":
        detector = DDM(**params)
    elif name_detector == "HDDM_A":
        detector = HDDM_A(**params)
    elif name_detector == "HDDM_W":
        detector = HDDM_W(**params)
    elif name_detector == "KSWIN":
        detector = KSWIN(**params)
    elif name_detector == "PageHinkley":
        detector = PageHinkley(**params)
    elif name_detector == "DataStream_Adapt":
        detector = DataStream_Adapt(**params)
    elif name_detector == "EWMAD_DT":
        detector = EWMAD_DT(**params)

    return detector


def max_metric(name_detector, params, true_drift_points, judge_size, model, name, X, all_Y, window_size, eta):
    time_list = []
    detector = set_params(name_detector, params)
    start_time = time.time()
    arch = architecture(model, name, X, all_Y, window_size, detector, name_detector, eta)
    drift_points, abrupt_drift_point, incremental_begin_points, incremental_end_points, outlier_points, res_nonotlier, res = arch.process()
    end_time = time.time()
    execution_time = end_time - start_time
    time_list.append(execution_time)
    # if name != "Ipod":
    #     drift_points, abrupt_drift_point, incremental_begin_points, incremental_end_points,res_nonotlier, res = drift_detection_with_detector(
    #         model, X, all_Y, window_size, detector, name_detector)
    #     outlier_points = []
    #     F1_Score_ = 0
    # elif name == "Ipod":
    #     print(params)
    #     ipcd = IPCD(X=X, Y=all_Y, window_size=window_size, detector=detector, detetor_name=name_detector, eta=eta,
    #                 outlier_test=True)
    #     drift_points, abrupt_drift_point, incremental_begin_points, incremental_end_points, outlier_points,res_nonotlier, res = ipcd.fit()
    #     # print(drift_points)
    true_outlier_points = outlier_list

    TP_ = 0
    FP_ = 0
    FN_ = 0
    for detect in outlier_points:
        # print(detect)
        # 检测到的点是否在实际漂移点的有效范围内
        if any(((detect - actual) == 0) for actual in
               true_outlier_points):
            TP_ += 1  # 真阳性
        else:
            FP_ += 1  # 假阳性
        # for actual in true_outlier_points:
        #     if (detected - actual) == 0 and (detected - actual) >= 0:
        #         # delay.append(detected - actual)
    # 漏报的漂移点
    for actual in true_outlier_points:
        if not any(((detect - actual) == 0) for detect in
                   outlier_points):
            FN_ += 1  # 假阴性
    # print(TP_, FP_, FN_)
    Precision_ = TP_ / (TP_ + FP_)
    Recall_ = TP_ / (TP_ + FN_)
    if TP_ != 0:
        F1_Score_ = 2 * (Precision_ * Recall_) / (Precision_ + Recall_)
    else:
        F1_Score_ = 0
    print(F1_Score_, Precision_, Recall_, TP_, FP_, FN_)
    # judge_size = 100  # 漂移前后50个点为有效范围
    # true_drift_points = [i for i in range(n_i, len(X), n_i)]  # 模拟真实漂移点

    F1_Score, Precision, Recall, TP, FP, FN, delay = F1_score(drift_points, true_drift_points, judge_size)
    print("alldrift", F1_Score, Precision, Recall, TP, FP, FN)

    # judge_size = 50  # 漂移前后50个点为有效范围
    if drift_type == "abrupt" or drift_type == "incremental":
        true_abrupt_points = [i for i in range(n_i, len(X), n_i)]  # 模拟真实漂移点
    elif drift_type == "mixed":
        true_abrupt_points = [n_i * i for i in mix_random_numbers]
    F1_Score_abrupt, Precision_abrupt, Recall_abrupt, TP_abrupt, FP_abrupt, FN_abrupt, delay_abrupt = F1_score(
        abrupt_drift_point, true_abrupt_points, judge_size)
    print("abrupt", F1_Score_abrupt, Precision_abrupt, Recall_abrupt, TP_abrupt, FP_abrupt, FN_abrupt)

    if drift_type == "abrupt" or drift_type == "incremental":
        true_incre_points = [i for i in range(n_i, len(X), n_i)]  # 模拟真实漂移点
    elif drift_type == "mixed":
        true_incre_points = [n_i * i for i in remaining]
    F1_Score_incre, Precision_incre, Recall_incre, TP_incre, FP_incre, FN_incre, delay_incre = F1_score(
        incremental_begin_points, true_incre_points, judge_size)
    print("incremental", F1_Score_incre, Precision_incre, Recall_incre, TP_incre, FP_incre, FN_incre)


    results = {
        "detector": name_detector,
        "Model": name,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": Precision,
        "Recall": Recall,
        "F1_Score_alldrift": F1_Score,
        "F1_Score_abrupt": F1_Score_abrupt,
        "F1_Score_incremental": F1_Score_incre,
        "F1_Score_outlier": F1_Score_,
        "time": np.mean(time_list),
        "mean_delay": np.mean(delay),
        "mean_delay_abrupt": np.mean(delay_abrupt),
        "mean_delay_incre": np.mean(delay_incre),
        "mean_res_nonotlier": np.mean(res_nonotlier),
        "mean_res": np.mean(res),
    }

    # if F1_Score >= best_score:
    #     best_score = F1_Score
    #     best_param = params
    #     best_results = results
    #     best_drift_points = drift_points
    #     best_outlier_points = outlier_points
    #     best_abrupt = abrupt_drift_point
    #     best_incre = incremental_begin_points
    #     best_incre_end = incremental_end_points

    return F1_Score, results, drift_points, outlier_points, abrupt_drift_point, incremental_begin_points, incremental_end_points, np.mean(
        res_nonotlier), np.mean(res)


class OptimizationWrapper:
    """包装类，用于保存优化所需的所有参数"""

    def __init__(self, name_detector, true_drift_points, judge_size,
                 model, name, X, all_Y, window_size, eta):
        self.name_detector = name_detector
        self.true_drift_points = true_drift_points
        self.judge_size = judge_size
        self.model = model
        self.name = name
        self.X = X
        self.all_Y = all_Y
        self.window_size = window_size
        self.eta = eta
        self.best_results = None
        self.best_score = np.inf
        self.best_param = None
        self.best_drift_points = []
        self.best_outlier_points = []
        self.best_abrupt = []
        self.best_incre = []
        self.best_incre_end = []
        self.best_res = 0

    def __call__(self, parameters):
        """使实例可调用，符合SeqUD的要求"""
        F1_Score, results, drift_points, outlier_points, abrupt_drift_point, incremental_begin_points, incremental_end_points, res_nonotlier, residual = max_metric(
            self.name_detector, parameters, self.true_drift_points, self.judge_size,
            self.model, self.name, self.X, self.all_Y, self.window_size, self.eta
        )

        # if F1_Score > self.best_score:
        #     self.best_score = F1_Score
        #     self.best_param = parameters
        #     self.best_results = results
        #     self.best_drift_points = drift_points
        #     self.best_outlier_points = outlier_points
        #     self.best_abrupt = abrupt_drift_point
        #     self.best_incre = incremental_begin_points
        #     self.best_incre_end = incremental_end_points

        if residual < self.best_score:
            self.best_score = residual
            self.best_param = parameters
            self.best_results = results
            self.best_drift_points = drift_points
            self.best_outlier_points = outlier_points
            self.best_abrupt = abrupt_drift_point
            self.best_incre = incremental_begin_points
            self.best_incre_end = incremental_end_points
            self.best_res = res_nonotlier

        return -residual


plt.rcParams['figure.figsize'] = (8, 10)
linestyles = ['-.', '--', ':', '-.', '--', '-', '-']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#FFB6C1', '#9acd32', '#eee8aa', '#8470ff', '#625b57', '#87cefa', '#f44336']

n1 = 30
# 定义每个检测器的参数网格
param_grids = {
    "ADWIN": {"delta": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x}},
    "DDM": {
        "min_num_instances": {'Type': 'integer', 'Mapping': list(range(1, n1))},
        "warning_level": {'Type': 'continuous', 'Range': [0, 2], 'Wrapper': lambda x: x},
        "out_control_level": {'Type': 'continuous', 'Range': [2, 4], 'Wrapper': lambda x: x}
    },
    "HDDM_W": {
        "drift_confidence": {'Type': 'continuous', 'Range': [0, 0.1], 'Wrapper': lambda x: x},
        "lambda_option": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
    },
    "KSWIN": {
        "alpha": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
        "window_size": {'Type': 'integer', 'Mapping': list(range(100, 300))},
        "stat_size": {'Type': 'integer', 'Mapping': list(range(1, n1))}
    },
    "PageHinkley": {
        "delta": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
        "threshold": {'Type': 'continuous', 'Range': [0, 20], 'Wrapper': lambda x: x},
        "burn_in": {'Type': 'integer', 'Mapping': list(range(1, n1))},
        "k": {'Type': 'integer', 'Mapping': list(range(1, n1))}
        # "direction": ["positive"]
    },
    "DataStream_Adapt": {
        "delta": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
        "threshold": {'Type': 'continuous', 'Range': [0, 10], 'Wrapper': lambda x: x},
        "burn_in": {'Type': 'integer', 'Mapping': list(range(1, n1))},
        "k": {'Type': 'integer', 'Mapping': list(range(1, n1))}
        # "direction": ["positive"]
    },
    "EWMAD_DT": {
        "threshold": {'Type': 'continuous', 'Range': [0.5, 1], 'Wrapper': lambda x: x},
        "burn_in": {'Type': 'integer', 'Mapping': list(range(1, n1))},
        "alpha": {'Type': 'continuous', 'Range': [0, 0.2], 'Wrapper': lambda x: x},
        "k": {'Type': 'integer', 'Mapping': list(range(1, n1))},
        # "direction": ["positive"]
    }
}

detectors = {
    # "ADWIN": ADWIN(delta=0.002),
    # "KSWIN": KSWIN(alpha=0.000001, window_size=100, stat_size=30, data=None),
    # "PageHinkley": PageHinkley(delta=0.001, threshold=5, burn_in=20, k=10, direction="positive"),
    # "DataStream_Adapt": DataStream_Adapt(delta=0.001, threshold=5, burn_in=20, k=10, direction="positive"),
    "EWMAD_DT": EWMAD_DT(threshold=0.8, burn_in=20, alpha=0.1, k=20, direction="positive")
}

# 模拟真实漂移点
true_drift_points = [i for i in range(n_i, len(X), n_i)]
F1score = np.ones((len(rate), 1, 8)) * (-1)
results_0 = {
    "detector": None,  # 检测器名称（字符串）
    "Model": None,  # 模型名称（字符串）
    "TP": 0,  # True Positives（整数）
    "FP": 0,  # False Positives（整数）
    "FN": 0,  # False Negatives（整数）
    "Precision": 0.0,  # 精确率（浮点数）
    "Recall": 0.0,  # 召回率（浮点数）
    "F1_Score_alldrift": 0.0,  # 漂移检测的 F1 分数（浮点数）
    "F1_Score_outlier": 0.0,  # 异常检测的 F1 分数（浮点数）
    "time": 0.0,  # 执行时间（浮点数，单位：秒）
    "mean_delay": float('inf'),  # 平均延迟（浮点数，初始设为无穷大）
    "mean_delay_abrupt": float('inf'),
    "mean_delay_incre": float('inf')
}
for num_r, r in enumerate(rate):
    # Define results path and create directory.
    path = './Results/'
    path += 'simu' + '_'
    path += drift_type + '_'
    path += "%s" % r + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # 生成数据
    # incremental
    random_integers = np.random.rand(d)
    beta_0 = np.array([
        [random_integers[0]],
        [random_integers[1]],
        [random_integers[2]],
        [random_integers[3]],
        [random_integers[4]],
        [random_integers[5]],
        [random_integers[6]],
        [random_integers[7]],
        [random_integers[8]],
        [random_integers[9]]
    ])
    current_beta = beta_0
    # abrupt
    all_Y = np.array([])
    for i in range(bins):
        # print(i)
        np.random.seed(i)
        # 生成四个随机整数
        random_integers = np.random.rand(d)  # 在1到100之间生成四个随机整数
        # print(i,random_integers)
        # 将这些整数赋值给 beta
        beta = np.array([
            [random_integers[0]],
            [random_integers[1]],
            [random_integers[2]],
            [random_integers[3]],
            [random_integers[4]],
            [random_integers[5]],
            [random_integers[6]],
            [random_integers[7]],
            [random_integers[8]],
            [random_integers[9]]
        ])
        if drift_type == "incremental":
            delta = (beta - current_beta) / incremental_length
            Y = np.zeros(n_i)
            for j in range(n_i):
                if j < incremental_length:
                    # 计算 Sigmoid 进度 (0 到 1 之间)
                    current_beta = current_beta + delta
                    # print(current_beta)
                # beta_evolution[j] = current_beta.flatten()
                # print(current_beta.shape)
                X_bin = X[i * n_i:(i + 1) * n_i, :]
                epsilon_bin = epsilon[i * n_i:(i + 1) * n_i]
                Y[j] = np.dot(X_bin[j], current_beta) + epsilon_bin[j]
                Y = Y.reshape(-1, 1)
            # print(Y)
        elif drift_type == "abrupt":
            current_beta = beta
            Y = np.dot(X[i * n_i:(i + 1) * n_i, :], current_beta) + epsilon[i * n_i:(i + 1) * n_i]
        elif drift_type == "mixed":
            if i in mix_random_numbers:
                current_beta = beta
                Y = np.dot(X[i * n_i:(i + 1) * n_i, :], current_beta) + epsilon[i * n_i:(i + 1) * n_i]
            else:
                delta = (beta - current_beta) / incremental_length
                Y = np.zeros(n_i)
                for j in range(n_i):
                    if j < incremental_length:
                        # 计算 Sigmoid 进度 (0 到 1 之间)
                        current_beta = current_beta + delta
                        # print(current_beta)
                    # beta_evolution[j] = current_beta.flatten()
                    # print(current_beta.shape)
                    X_bin = X[i * n_i:(i + 1) * n_i, :]
                    epsilon_bin = epsilon[i * n_i:(i + 1) * n_i]
                    Y[j] = np.dot(X_bin[j], current_beta) + epsilon_bin[j]
                    Y = Y.reshape(-1, 1)

        # 生成范围在 (-3, -2) ∪ (2, 3) 上的随机数
        # for outlier in range(4):
        #     random_number = np.random.uniform(low=-3, high=-2, size=1)  # 在区间 (-3, -2) 上生成一个随机数
        #     if np.random.rand() < 0.5:  # 以50%的概率选择下一个区间
        #         random_number = np.random.uniform(low=2, high=3, size=1)  # 在区间 (2, 3) 上生成一个随机数
        outlier_list = []

        # print(outlier_list)
        for outlier in range(int(n * r / bins)):
            mean = np.random.uniform(low=0.5, high=1, size=1)  # 在区间 (2, 3) 上生成一个随机数
            std_dev = np.random.uniform(low=0, high=0.1, size=1)  # 在区间 (2, 3) 上生成一个随机数
            random_number = np.random.normal(mean, std_dev, size=1)  # 在区间 (-3, -2) 上生成一个随机数
            if np.random.rand() < 0.5:  # 以50%的概率选择下一个区间
                random_number = np.random.normal((-mean), std_dev, size=1)  # 在区间 (2, 3) 上生成一个随机数

            # print("Random number in (-3, -2) ∪ (2, 3):", random_number)
            outlier_index = int(n / (bins * (int(n * r / bins) + 1)) * (outlier + 1))
            # outlier_index = 1/r  * (outlier + 1)
            # outlier_list.append(outlier_index)
            # print(outlier_index)
            if outlier_index < len(Y):
                Y[int(outlier_index)] = Y[int(outlier_index)] + random_number[0]

            for i in range(bins):
                outlier_list.append(n_i * i + outlier_index)
            # print(outlier_list)
        # 拼接Y到all_Y
        if all_Y.size == 0:
            all_Y = Y
        else:
            all_Y = np.concatenate((all_Y, Y))
        # print(all_Y.shape)
    # 对threshold循环
    # for num_thres,thres in enumerate(threshold):
    for num_eta, eta in enumerate(eta_list):
        results_list = []
        best_params_all = []
        for num_detector, (name_detector, detector) in enumerate(detectors.items()):
            print(name_detector)
            # 模型字典
            models = {
                "RANSACRegressor": RANSACRegressor(estimator=LinearRegression(), min_samples=2, max_trials=100,
                                                   stop_probability=0.99, random_state=42),
                "HuberRegressor": HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000, tol=1e-5),
                "TheilSenRegressor": TheilSenRegressor(max_subpopulation=1e3, random_state=42),
                "$\Theta$-IPOD": IPCD(X=X, Y=all_Y, window_size=window_size, detector=detector,
                                      detetor_name=name_detector,
                                      eta=eta),
                "ARLF": FastAdaptiveRobustRegressor(input_dim=10, epochs=30, learning_rate=0.1),
                # "ARTL":Fast_TTL_HOVR_Regressor(input_dim=10,epochs=30, learning_rate=0.1),
                # "Linear": LinearRegression(),
            }

            # 循环遍历每个模型
            # results_list = []
            # time_list = []
            for num, (name, model) in enumerate(models.items()):
                print(f"Processing model: {name}")
                ParaSpace = param_grids[name_detector]
                clf = SeqUD(ParaSpace, max_runs=50, random_state=2, verbose=True)
                optimizer = OptimizationWrapper(
                    name_detector, true_drift_points, judge_size,
                    model, name, X, all_Y, window_size, eta
                )
                clf.fmax(optimizer)

                results_list.append(optimizer.best_results)
                print(num_detector, num_r, num)
                print(optimizer.best_results)
                F1score[num_r, 0, num] = optimizer.best_score
                best_params_all.append(optimizer.best_param)

                if name_detector == "EWMAD_DT" or name_detector == "DataStream_Adapt":
                    plt.figure(figsize=(12, 6))
                    plt.plot(X, color="purple", label="X")
                    plt.plot(all_Y, color="black", label="Y")
                    for dp in optimizer.best_abrupt:
                        plt.axvline(dp, color="red", linestyle="--",
                                    label="Detected Abrupt Drift" if dp == optimizer.best_abrupt[0] else "")
                    for dp in optimizer.best_incre:
                        plt.axvline(dp, color="pink", linestyle="--",
                                    label="Detected Incremental Drift" if dp == optimizer.best_incre[0] else "")
                    # for dp in best_incre_end:
                    #     plt.axvline(dp, color="orange", linestyle="--", label="Detected Incremental end" if dp == best_incre_end[0] else "")
                    for op in optimizer.best_outlier_points:
                        plt.axvline(op, color="green", linestyle="--",
                                    label="Detected Outlier" if op == optimizer.best_outlier_points[0] else "")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.title(f"{name} and {name_detector} Drift Detection_simu")
                    if name == "$\Theta$-IPOD":
                        plt.savefig(f"{path}/IPOD_and_{name_detector}_Drift_Detection_simu.png", bbox_inches="tight")
                    else:
                        plt.savefig(f"{path}/{name}_and_{name_detector}_Drift_Detection_simu.png", bbox_inches="tight")
                    # plt.show()
                else:
                    plt.figure(figsize=(12, 6))
                    plt.plot(X, color="purple", label="X")
                    plt.plot(all_Y, color="black", label="Y")
                    for dp in optimizer.best_drift_points:
                        plt.axvline(dp, color="orange", linestyle="--",
                                    label="Detected Drift" if dp == optimizer.best_drift_points[0] else "")
                    for op in optimizer.best_outlier_points:
                        plt.axvline(op, color="green", linestyle="--",
                                    label="Detected Outlier" if op == optimizer.best_outlier_points[0] else "")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.title(f"{name} and {name_detector} Drift Detection_simu")
                    if name == "$\Theta$-IPOD":
                        plt.savefig(f"{path}/IPOD_and_{name_detector}_Drift_Detection_simu.png", bbox_inches="tight")
                    else:
                        plt.savefig(f"{path}/{name}_and_{name_detector}_Drift_Detection_simu.png", bbox_inches="tight")
                    # plt.show()

        df_results = pd.DataFrame(results_list)
        # 创建全局参数表格
        global_params = {
            "Parameter": ["window_size", "judge_size", "X_dim", "rate", "eta"],
            "Value": [window_size, judge_size, X.shape[1], r, eta]
        }
        df_global_params = pd.DataFrame(global_params)
        df_params = pd.DataFrame(best_params_all)

        # 保存到 Excel
        with pd.ExcelWriter(f"%s/out_simu_{drift_type}_rate_{r:.3f}_{eta:.4f}.xlsx" % path) as writer:
            df_global_params.to_excel(writer, sheet_name="Global Parameters", index=False)
            df_results.to_excel(writer, sheet_name="Model Results", index=False)
            df_params.to_excel(writer, sheet_name="Best Params", index=False)
