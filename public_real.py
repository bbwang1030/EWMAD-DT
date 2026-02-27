import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from detetor import PageHinkley, DDM, HDDM_A, KSWIN,ADWIN, HDDM_W,DDM_
from DataStream_Adapt import DataStream_Adapt
from EWMAD_DT import EWMAD_DT
import pandas as pd
from pyoselm.oselm import OSELMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,HuberRegressor,RANSACRegressor,TheilSenRegressor
from IPCD import IPCD
import time
from sklearn.model_selection import ParameterGrid
from sequd import SeqUD
import os
from IPCD import *
from robust_NN_real_data import *
import copy
# 数据生成
np.random.seed(2)

# datasets = ["micro"]
# datasets = ["bike_day"]
# datasets = ["worker"]
# datasets = ["Metro"]
# datasets = ["energy","Metro"]
# datasets = ["Airquality"]
# datasets = ["pi4","pi5"]
# datasets = ["worker","Airquality","bike_day","pi4","pi2","pi3","pi5"]
# datasets = ["pi2","pi3"]
datasets = ["worker","Airquality","bike_day"]

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

        # 结果记录
        self.drift_points = []
        self.outlier_points = []
        self.abrupt_drift_point = []
        self.abrupt_drift_point_pre = []
        self.incremental_begin_points = []
        self.incremental_end_points = []

        # 状态标志
        self.outlier_warning = "not_outlier_warning"
        self.outlier_warning_before = False
        self.outlier_check_before = False
        self.drift_state_before = None
        self.pending_drift_idx = None

        # === 新增/修改的核心变量 ===
        self.beta_ols = np.zeros((X.shape[1],1))  # 用于存储 IPOD 模型的参数
        self.mean_res = 0  # 当前模型的残差均值
        self.std_res = 1  # 当前模型的残差标准差

        # 缓冲机制变量
        self.is_buffering = False
        self.buffer_X = []
        self.buffer_Y = []
        self.buffer_residuals = []
    def _retrain_model(self, X_train, Y_train):
        """
        辅助函数：使用给定的数据训练模型，并更新 mean/std 统计量
        """
        if self.model_name == "$\Theta$-IPOD":
            # 计算 Hat Matrix
            reference_H = np.dot(
                np.dot(X_train, inv(np.dot(X_train.T, X_train))),
                X_train.T
            )
            # 调用 IPOD 算法
            # print(1111)
            reference_result = IPOD_new(X_train, Y_train, reference_H, eta=self.eta)
            # print(2222)
            gamma = reference_result["gamma"].reshape((len(X_train), 1))

            # 更新 OLS 参数 (self.beta_ols)
            self.beta_ols = np.dot(
                inv(np.dot(X_train.T, X_train)),
                np.dot(X_train.T, Y_train - gamma)
            )

            # 更新统计量
            residuals = Y_train - np.dot(X_train, self.beta_ols) - gamma
            self.mean_res = np.mean(residuals)
            self.std_res = np.std(residuals)

        else:
            # 普通模型训练
            self.model.fit(X_train, Y_train.ravel())

            # 更新统计量
            preds = self.model.predict(X_train)
            residuals = Y_train - preds
            self.mean_res = np.mean(residuals)
            self.std_res = np.std(residuals)

    def process(self):
        res = []  # 存储所有残差
        res_nonotlier = []
        temp_warning_before = False
        temp_check_before = False

        for i in range(len(self.X)):

            # 获取当前单个数据点 (Current Instance)
            if self.model_name == "$\Theta$-IPOD":
                curr_X = self.X[i].reshape(1, -1)  # 保持维度一致
                curr_Y = self.Y[i]
            else:
                curr_X = self.X[i].reshape(1, -1)
                curr_Y = self.Y[i]

            # -------------------------------------------------------
            # A. 预测 (Prediction) - 始终使用“当前可用模型”
            # -------------------------------------------------------
            if self.model_name == "$\Theta$-IPOD":
                # 注意：curr_X 这里如果是 (1, features)，beta_ols 是 (features, 1)
                y_pred = np.dot(curr_X, self.beta_ols)[0]
            else:
                if i==0:
                    X_init = np.random.rand(window_size, X.shape[1])
                    Y_init = np.random.rand(window_size)  # 或者是 zeros

                    # 然后调用
                    self._retrain_model(X_init, Y_init)
                y_pred = self.model.predict(curr_X)[0]

            # 计算当前残差
            if isinstance(y_pred, (np.ndarray, list)):
                y_pred = y_pred[0]  # 确保是标量

            # 防止除以0错误，加个极小值
            denom = curr_Y
            residual = abs((curr_Y - y_pred) / denom)
            if self.is_buffering==False:
                res.append(float(residual))

            # 异常值检测 (基于当前模型的统计量)
            # 注意：residual 是绝对百分比误差，下面的 mean/std 应该是基于绝对误差还是什么？
            # 假设你的 self.mean_res 和 self.std_res 也是基于同一种误差计算的
            # 这里沿用你原来的逻辑： abs(new_Y - y_pred) > ...
            abs_diff = abs(curr_Y - y_pred)
            outlier_warning = abs(abs_diff-self.mean_res) >  (2 * self.std_res)
            outlier_check = abs(abs_diff-self.mean_res) >  (2.6 * self.std_res)

            # -------------------------------------------------------
            # B. 状态分支：缓冲期 vs 监控期
            # -------------------------------------------------------

            if self.is_buffering or i<self.window_size:
                # === 缓冲期 (Buffering Phase) ===
                # 即使在缓冲，我们已经做了预测(步骤A)，现在只负责收集数据
                # 不更新检测器！
                if self.model_name == "$\Theta$-IPOD":
                    curr_X = self.X[i].reshape(1, -1)
                    # print(curr_X.shape,self.beta_ols.shape)
                    y_pred = np.dot(curr_X, self.beta_ols)[0]
                else:
                    curr_X = self.X[i].reshape(1, -1)
                    y_pred = self.model.predict(curr_X)[0]

                curr_Y = self.Y[i]
                # 存绝对误差用于计算分布，同时也存原始值用于后续可能的训练
                raw_abs_diff = abs(curr_Y - y_pred)
                self.buffer_residuals.append(raw_abs_diff)
                self.buffer_X.append(self.X[i])
                self.buffer_Y.append(self.Y[i])

                # 1. 计算局部均值 (这个值很大，比如 10.0)
                local_mean = abs(np.mean(self.buffer_residuals))

                # 2. 计算局部波动 (这个值很小，比如 0.2)
                local_std = abs(np.std(self.buffer_residuals))

                outlier_warning_ = abs(raw_abs_diff-local_mean) >  (2 * local_std)
                outlier_check_ = abs(raw_abs_diff-local_mean) >  (2.6 * local_std)
                # print("i,raw_abs_diff", i, raw_abs_diff,local_mean)
                if temp_warning_before and not outlier_warning_ and len(self.buffer_residuals) >= 10:
                    # 警告消失，说明上一个点可能是 outlier (或者 warning 只是误报)
                    # 你的逻辑：Warning -> Normal (Warning消失) -> 检查之前是不是 Check 级别
                    if temp_check_before:
                        # 确认是 Outlier
                        # 还原全局索引
                        self.outlier_points.append(i-1)
                        del self.buffer_residuals[-2]


                temp_warning_before = outlier_warning_
                temp_check_before = outlier_check_

                # self.buffer_X.append(self.X[i])
                # self.buffer_Y.append(self.Y[i])

                # 检查缓冲区是否满了
                if len(self.buffer_X) == self.window_size:
                    # 缓冲区已满 -> 训练新模型
                    X_buf_arr = np.array(self.buffer_X)
                    Y_buf_arr = np.array(self.buffer_Y)

                    self._retrain_model(X_buf_arr, Y_buf_arr)
                    # print(i-1)
                    # 重置状态
                    # self.detector.reset()
                    self.is_buffering = False
                    self.buffer_X = []  # 清空
                    self.buffer_Y = []
                    self.buffer_residuals = []
                    # 记录漂移结束点（可选）
                    # self.incremental_end_points.append(i - self.window_size)

            else:
                # === 监控期 (Monitoring Phase) ===
                # 只有在非缓冲期，才更新检测器寻找漂移

                # 更新检测器逻辑 (沿用你的 Outlier 过滤逻辑)
                # print(i-1,res[-2],i,res[-1],abs_diff , self.mean_res+(2 * self.std_res))
                if self.outlier_warning_before and not outlier_warning and self.drift_state_before != "drift":
                    self.outlier_warning = "outlier_warning"
                    if self.outlier_check_before:
                        self.outlier_warning = "outlier"
                        self.outlier_points.append(i - 1)
                else:
                    if self.detector.samples_since_reset == 0:
                        self.detector.update(np.array(0))
                        res_nonotlier.append(0)
                    else:
                        # 使用前一个点的残差更新 (根据你的原始逻辑 res[-2])
                        # 注意：现在 res[-1] 是当前点，res[-2] 是上一个点
                        update_val = res[-2]
                        self.detector.update(np.array([update_val]))

                        # 如果不是漂移也不是 Outlier，记录为正常点
                        if not outlier_warning:
                            res_nonotlier.append(update_val)

                    # 检查是否发生漂移
                    if self.detector.drift_state == "drift":
                        # === 触发漂移 ===
                        # 1. 记录漂移点
                        current_idx = i - 1  # 保持和你原来索引一致
                        self.drift_points.append(current_idx)

                        # 处理具体的漂移类型记录 (EWMAD 等)
                        self._handle_drift_logging(current_idx)
                        # print(current_idx, self.detector.drift_state_type )
                        # 2. 开启缓冲模式
                        self.is_buffering = True
                        self.detector.reset()
                        # 3. 将**当前点**作为缓冲区的第一个点
                        self.buffer_X.append(self.X[i])
                        self.buffer_Y.append(self.Y[i])


                # 更新历史状态
                self.outlier_warning_before = outlier_warning
                self.outlier_check_before = outlier_check
                self.drift_state_before = self.detector.drift_state # 注意：buffer期间 drift_state 不会变
        # print("outlier_points",self.outlier_points)
        # print("drift_points",self.drift_points)
        return self.drift_points, self.abrupt_drift_point, self.incremental_begin_points, self.incremental_end_points, self.outlier_points, res_nonotlier, res

    def _handle_drift_logging(self, current_idx):
        """处理不同检测器的漂移点记录逻辑 (从原代码提取)"""
        distinguish = ["EWMAD_DT", "DataStream_Adapt"]
        non_distinguish = ["DDM", "PageHinkley", "HDDM_A", "HDDM_W", "KSWIN", "ADWIN"]

        if self.detetor_name == "EWMAD_DT":
            if self.detector.drift_state_type != "incremental":
                self.abrupt_drift_point.append(current_idx)
            elif self.detector.drift_state_type == "incremental":
                self.incremental_begin_points.append(current_idx)
            # self.drift_points = self.abrupt_drift_point + self.incremental_begin_points
            self.incremental_end_points = []

        elif self.detetor_name == "DataStream_Adapt":
            if self.detector.drift_state_type != "incremental":
                self.abrupt_drift_point.append(current_idx)
            elif self.detector.drift_state_type == "incremental":
                self.incremental_begin_points.append(current_idx)
            # self.drift_points = self.abrupt_drift_point + self.incremental_begin_points
            self.incremental_end_points = []

        elif self.detetor_name in non_distinguish:
            # 简单的漂移记录
            # 注意：我们在主循环里已经 append 到 self.drift_points 了
            # 这里清理其他列表
            self.incremental_end_points = []
            self.abrupt_drift_point = []
            self.incremental_begin_points = []


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



def F1_score(drift_points, true_drift_points, judge_size):
    # judge_size = 100  # 漂移前后50个点为有效范围
    # true_drift_points = [i for i in range(2000, len(X), 2000)]  # 模拟真实漂移点
    # 统计 TP, FP, FN
    TP = 0
    FP = 0
    FN = 0
    delay = []
    for detected in drift_points:
        # 检测到的点是否在实际漂移点的有效范围内
        if any(((detected - actual) <= judge_size and (detected - actual) >= 0) for actual in true_drift_points):
            TP += 1  # 真阳性
            # print(detected,true_drift_points)
        else:
            FP += 1  # 假阳性
        for actual in true_drift_points:
            if (detected - actual) <= judge_size and (detected - actual) >= 0:
                delay.append(detected - actual)
    # 漏报的漂移点
    for actual in true_drift_points:
        if not any(((detected - actual) <= judge_size and (detected - actual) >= 0) for detected in drift_points):
            FN += 1  # 假阴性

    # # Precision 和 Recall
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)

    if TP != 0:
        # Precision 和 Recall
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        F1_Score = 0
        Precision = 0
        Recall = 0

    return F1_Score, Precision, Recall, TP, FP, FN, delay


def max_metric(name_detector, params, true_drift_points, judge_size, model, name, X, all_Y, window_size, eta):
    # try:
    #     current_model = clone(model)
    # except:
    #     # 对于自定义模型 ARLF，如果没有实现 get_params，clone 会报错
    #     # 需要手动重新实例化，或者确保 ARLF 实现了 sklearn 接口
    #     # 简单粗暴法：利用 type(model) 重新创建
    #     current_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else copy.deepcopy(model)
    time_list = []
    detector = set_params(name_detector, params)
    start_time = time.time()
    arch = architecture(model, name, X, all_Y, window_size, detector, name_detector, eta)
    drift_points, abrupt_drift_point, incremental_begin_points, incremental_end_points, outlier_points, res_nonotlier, res = arch.process()
    end_time = time.time()
    execution_time = end_time - start_time
    time_list.append(execution_time)
    percentage = (len(res)-len(res_nonotlier))/len(all_Y)
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
    Precision_ = TP_ / (TP_ + FP_) if (TP_ + FP_) > 0 else 0
    Recall_ = TP_ / (TP_ + FN_) if (TP_ + FN_) > 0 else 0
    if TP_ != 0:
        F1_Score_ = 2 * (Precision_ * Recall_) / (Precision_ + Recall_)
    else:
        F1_Score_ = 0
    # print("outlier",F1_Score_, Precision_, Recall_, TP_, FP_, FN_)
    # judge_size = 100  # 漂移前后50个点为有效范围
    # true_drift_points = [i for i in range(n_i, len(X), n_i)]  # 模拟真实漂移点

    F1_Score, Precision, Recall, TP, FP, FN, delay = F1_score(drift_points, true_drift_points, judge_size)
    # print("alldrift", F1_Score, Precision, Recall, TP, FP, FN)

    # # judge_size = 50  # 漂移前后50个点为有效范围
    # if drift_type == "abrupt" or drift_type == "incremental":
    #     true_abrupt_points = [i for i in range(n_i, len(X), n_i)]  # 模拟真实漂移点
    # elif drift_type == "mixed":
    #     true_abrupt_points = [n_i * i for i in mix_random_numbers]
    # F1_Score_abrupt, Precision_abrupt, Recall_abrupt, TP_abrupt, FP_abrupt, FN_abrupt, delay_abrupt = F1_score(
    #     abrupt_drift_point, true_abrupt_points, judge_size)
    # print("abrupt", F1_Score_abrupt, Precision_abrupt, Recall_abrupt, TP_abrupt, FP_abrupt, FN_abrupt)
    #
    # if drift_type == "abrupt" or drift_type == "incremental":
    #     true_incre_points = [i for i in range(n_i, len(X), n_i)]  # 模拟真实漂移点
    # elif drift_type == "mixed":
    #     true_incre_points = [n_i * i for i in remaining]
    # F1_Score_incre, Precision_incre, Recall_incre, TP_incre, FP_incre, FN_incre, delay_incre = F1_score(
    #     incremental_begin_points, true_incre_points, judge_size)
    # print("incremental", F1_Score_incre, Precision_incre, Recall_incre, TP_incre, FP_incre, FN_incre)


    results = {
        "detector": name_detector,
        "Model": name,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": Precision,
        "Recall": Recall,
        "F1_Score_alldrift": F1_Score,
        # "F1_Score_abrupt": F1_Score_abrupt,
        # "F1_Score_incremental": F1_Score_incre,
        "F1_Score_outlier": F1_Score_,
        "time": np.mean(time_list),
        # "mean_delay": np.mean(delay)+1,
        # "mean_delay_abrupt": np.mean(delay_abrupt)+1,
        # "mean_delay_incre": np.mean(delay_incre)+1,
        "mean_res_nonotlier": np.mean(res_nonotlier),
        "mean_res": np.mean(res),
        "percentage_delate":percentage
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


for dataset_name in datasets:
    if dataset_name == "pi4":
        window_size = 150
        eta = 1 / 20
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        n1=50
        n2=30

        data = pd.read_excel('./Dataset/pi4.xlsx')
        data.replace('None', np.nan, inplace=True)

        # Remove rows containing NaN values
        data.dropna(subset=['Humidity', 'Temperature'], inplace=True)

        # Convert data to numeric type
        data['Humidity'] = pd.to_numeric(data['Humidity'], errors='coerce')
        data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')

        # Remove NaN values introduced by conversion
        data.dropna(subset=['Humidity', 'Temperature'], inplace=True)

        # Define Y and X
        X = data['Humidity']
        Y = data['Temperature']
        X = X.values.reshape(-1, 1)
        all_Y = Y.values.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = [24, 210, 411, 456, 517, 810, 877, 940, 1130, 1541, 1597, 1617, 1659]
        true_abrupt_points = [24, 810, 1597, 1617, 1659]
        true_incre_points = [210, 411, 456, 517, 877, 940, 1130, 1541, 1562]
        outlier_list = [768, 962, 1688]
    elif dataset_name == "pi5" or dataset_name == "pi2" or dataset_name == "pi3":
        window_size = 100
        eta = 1 / 20
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        n1=50
        n2=30

        if dataset_name == "pi5":
            data = pd.read_excel('./Dataset/pi5.xlsx')
        elif dataset_name == "pi2":
            data = pd.read_excel('./Dataset/pi2.xlsx')
        elif dataset_name == "pi3":
            data = pd.read_excel('./Dataset/pi3.xlsx')

        data.replace('None', np.nan, inplace=True)

        # Remove rows containing NaN values
        data.dropna(subset=['Humidity', 'Temperature'], inplace=True)

        # Convert data to numeric type
        data['Humidity'] = pd.to_numeric(data['Humidity'], errors='coerce')
        data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')

        # Remove NaN values introduced by conversion
        data.dropna(subset=['Humidity', 'Temperature'], inplace=True)

        # Define Y and X
        X = data['Humidity']
        Y = data['Temperature']
        X = X.values.reshape(-1, 1)
        all_Y = Y.values.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = [24, 210, 411, 456, 517, 810, 877, 940, 1130, 1541, 1597, 1617, 1659]
        true_abrupt_points = [24, 810, 1597, 1617, 1659]
        true_incre_points = [210, 411, 456, 517, 877, 940, 1130, 1541, 1562]
        outlier_list = [768, 962, 1688]
    elif dataset_name == "bike":
        window_size = 50
        eta = 1 / 8
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        # rate = [0]
        n1=30
        n2=30

        data = pd.read_excel('./Dataset/BikeDrift.xlsx')
        data.replace('None', np.nan, inplace=True)

        # Remove rows containing NaN values
        data.dropna(subset=['x_holiday', 'x_weekday','x_workingday', 'x_weathersit','x_temp', 'x_atemp','x_hum', 'x_windspeed','y'], inplace=True)

        # Convert data to numeric type
        data['x_temp'] = pd.to_numeric(data['x_temp'], errors='coerce')
        data['x_atemp'] = pd.to_numeric(data['x_atemp'], errors='coerce')
        data['x_hum'] = pd.to_numeric(data['x_hum'], errors='coerce')
        data['x_windspeed'] = pd.to_numeric(data['x_windspeed'], errors='coerce')
        data['y'] = pd.to_numeric(data['y'], errors='coerce')

        # Define Y and X
        X = data[['x_temp','x_atemp','x_hum','x_windspeed']].values
        Y = data['y'].values
        # X = X.values.reshape(-1, 1)
        all_Y = Y.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = []
        true_abrupt_points = []
        true_incre_points = []
        outlier_list = []
    elif dataset_name == "bike_day":
        window_size = 50
        eta = 1 / 8
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        # rate = [0]
        n1=20
        n2=20

        data = pd.read_excel('./Dataset/bike_day.xlsx')
        data.replace('None', np.nan, inplace=True)

        # Remove rows containing NaN values
        data.dropna(subset=['holiday', 'weekday','weekday', 'weathersit','temp', 'atemp','hum', 'windspeed','casual','registered','cnt'], inplace=True)

        # Convert data to numeric type
        data['x_temp'] = pd.to_numeric(data['temp'], errors='coerce')
        data['x_atemp'] = pd.to_numeric(data['atemp'], errors='coerce')
        data['x_hum'] = pd.to_numeric(data['hum'], errors='coerce')
        data['x_windspeed'] = pd.to_numeric(data['windspeed'], errors='coerce')
        data['x_casual'] = pd.to_numeric(data['casual'], errors='coerce')
        data['x_registered'] = pd.to_numeric(data['registered'], errors='coerce')
        data['y'] = pd.to_numeric(data['cnt'], errors='coerce')

        # Define Y and X
        X = data[['x_temp','x_atemp','x_hum','x_windspeed']].values
        Y = data['y'].values
        # X = X.values.reshape(-1, 1)
        all_Y = Y.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = []
        true_abrupt_points = []
        true_incre_points = []
        outlier_list = []
    elif dataset_name == 'worker':
        window_size = 30
        eta = 1 / 8
        judge_size = 20  # 漂移前后30个点为有效范围
        rate = [0]
        # rate = [0]
        n1 = 20
        n2 = 20

        data = pd.read_excel('./Dataset/worker_productivity.xlsx')
        data.replace('None', np.nan, inplace=True)

        # Remove rows containing NaN values
        data.dropna(
            subset=['smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
                    'no_of_workers', 'actual_productivity'], inplace=True)

        # Convert data to numeric type
        data['team'] = pd.to_numeric(data['team'], errors='coerce')
        data['targeted_productivity'] = pd.to_numeric(data['targeted_productivity'], errors='coerce')
        data['smv'] = pd.to_numeric(data['smv'], errors='coerce')
        data['wip'] = pd.to_numeric(data['wip'], errors='coerce')
        data['over_time'] = pd.to_numeric(data['over_time'], errors='coerce')
        data['incentive'] = pd.to_numeric(data['incentive'], errors='coerce')
        data['idle_time'] = pd.to_numeric(data['idle_time'], errors='coerce')
        data['idle_men'] = pd.to_numeric(data['idle_men'], errors='coerce')
        data['no_of_style_change'] = pd.to_numeric(data['no_of_style_change'], errors='coerce')
        data['no_of_workers'] = pd.to_numeric(data['no_of_workers'], errors='coerce')
        data['actual_productivity'] = pd.to_numeric(data['actual_productivity'], errors='coerce')


        # Define Y and X
        X = data[[ 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive',
                  'no_of_workers']].values
        Y = data['actual_productivity'].values
        # X = X.values.reshape(-1, 1)
        all_Y = Y.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)
        print(len(all_Y))

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 50)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = []
        true_abrupt_points = []
        true_incre_points = []
        outlier_list = []
    elif dataset_name == "Metro":
        window_size = 50
        eta = 1 / 8
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        # rate = [0]
        n1 = 20
        n2 = 20

        data = pd.read_excel('./Dataset/Metro.xlsx')
        data.replace('None', np.nan, inplace=True)
        # print(data)
        # Remove rows containing NaN values
        # data.dropna(
        #     subset=['ISE1','ISE', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA','EU', 'EM'], inplace=True)

        # Convert data to numeric type
        data['temp'] = pd.to_numeric(data['temp'], errors='coerce')
        data['clouds_all'] = pd.to_numeric(data['clouds_all'], errors='coerce')
        data['traffic_volume'] = pd.to_numeric(data['traffic_volume'], errors='coerce')

        # Define Y and X
        X = data[['temp']].values
        Y = data['traffic_volume'].values
        # X = X.values.reshape(-1, 1)
        all_Y = Y.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)
        print(len(all_Y))

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = []
        true_abrupt_points = []
        true_incre_points = []
        outlier_list = []
    elif dataset_name == "energy":
        window_size = 500
        eta = 1 / 8
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        # rate = [0]
        n1 = 20
        n2 = 20

        data = pd.read_excel('./Dataset/energydata_1.xlsx')
        data.replace('None', np.nan, inplace=True)
        # print(data)
        # Remove rows containing NaN values
        data.dropna(
            subset=['lights','T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7',
                  'T8','RH_8','T9','RH_9','T_out','RH_out','Press_mm_hg','Windspeed','Visibility','Tdewpoint','rv1','Appliances'], inplace=True)

        # Convert data to numeric type
        data['Appliances'] = pd.to_numeric(data['Appliances'], errors='coerce')
        data['lights'] = pd.to_numeric(data['lights'], errors='coerce')
        data['T1'] = pd.to_numeric(data['T1'], errors='coerce')
        data['RH_1'] = pd.to_numeric(data['RH_1'], errors='coerce')
        data['T2'] = pd.to_numeric(data['T2'], errors='coerce')
        data['RH_2'] = pd.to_numeric(data['RH_2'], errors='coerce')
        data['T3'] = pd.to_numeric(data['T3'], errors='coerce')
        data['RH_3'] = pd.to_numeric(data['RH_3'], errors='coerce')
        data['T4'] = pd.to_numeric(data['T4'], errors='coerce')
        data['RH_4'] = pd.to_numeric(data['RH_4'], errors='coerce')
        data['T5'] = pd.to_numeric(data['T5'], errors='coerce')
        data['RH_5'] = pd.to_numeric(data['RH_5'], errors='coerce')
        data['T6'] = pd.to_numeric(data['T6'], errors='coerce')
        data['RH_6'] = pd.to_numeric(data['RH_6'], errors='coerce')
        data['T7'] = pd.to_numeric(data['T7'], errors='coerce')
        data['RH_7'] = pd.to_numeric(data['RH_7'], errors='coerce')
        data['T8'] = pd.to_numeric(data['T8'], errors='coerce')
        data['RH_8'] = pd.to_numeric(data['RH_8'], errors='coerce')
        data['T9'] = pd.to_numeric(data['T9'], errors='coerce')
        data['RH_9'] = pd.to_numeric(data['RH_9'], errors='coerce')
        data['RH_out'] = pd.to_numeric(data['RH_out'], errors='coerce')
        data['T_out'] = pd.to_numeric(data['T_out'], errors='coerce')
        data['Press_mm_hg'] = pd.to_numeric(data['Press_mm_hg'], errors='coerce')
        data['Windspeed'] = pd.to_numeric(data['Windspeed'], errors='coerce')
        data['Visibility'] = pd.to_numeric(data['Visibility'], errors='coerce')
        data['Tdewpoint'] = pd.to_numeric(data['Tdewpoint'], errors='coerce')
        data['rv1'] = pd.to_numeric(data['rv1'], errors='coerce')

        # Define Y and X
        X = data[['RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7',
                  'RH_8','RH_9','T_out','RH_out','Press_mm_hg','Windspeed','Tdewpoint']].values
        Y = data['Appliances'].values
        # X = X.values.reshape(-1, 1)
        all_Y = Y.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)
        print(len(all_Y))

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = []
        true_abrupt_points = []
        true_incre_points = []
        outlier_list = []
    elif dataset_name == "Airquality":
        window_size = 50
        eta = 1 / 8
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0]
        # rate = [0]
        n1 = 20
        n2 = 20

        data = pd.read_excel('./Dataset/AirQualityUCI.xlsx')
        data.replace('None', np.nan, inplace=True)
        # print(data)
        # Remove rows containing NaN values
        data.dropna(
            subset=['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)',
                    'NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH',], inplace=True)

        # Convert data to numeric type
        data['CO(GT)'] = pd.to_numeric(data['CO(GT)'], errors='coerce')
        data['PT08.S1(CO)'] = pd.to_numeric(data['PT08.S1(CO)'], errors='coerce')
        data['NMHC(GT)'] = pd.to_numeric(data['NMHC(GT)'], errors='coerce')
        data['C6H6(GT)'] = pd.to_numeric(data['C6H6(GT)'], errors='coerce')
        data['PT08.S2(NMHC)'] = pd.to_numeric(data['PT08.S2(NMHC)'], errors='coerce')
        data['NOx(GT)'] = pd.to_numeric(data['NOx(GT)'], errors='coerce')
        data['PT08.S3(NOx)'] = pd.to_numeric(data['PT08.S3(NOx)'], errors='coerce')
        data['NO2(GT)'] = pd.to_numeric(data['NO2(GT)'], errors='coerce')
        data['PT08.S4(NO2)'] = pd.to_numeric(data['PT08.S4(NO2)'], errors='coerce')
        data['PT08.S5(O3)'] = pd.to_numeric(data['PT08.S5(O3)'], errors='coerce')
        data['T'] = pd.to_numeric(data['T'], errors='coerce')
        data['RH'] = pd.to_numeric(data['RH'], errors='coerce')
        data['AH'] = pd.to_numeric(data['AH'], errors='coerce')

        # Define Y and X
        X = data[['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)',
                  'PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']].values
        Y = data['NO2(GT)'].values
        # X = X.values.reshape(-1, 1)
        all_Y = Y.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)
        print(len(all_Y))

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        # 模拟真实漂移点
        # true_drift_points = [55,81,121,187, 250, 304,333,412,476,537, 557,648, 745, 804,826,873,918,945,997,1032,1069,1090,1112,1153,1190, 1217, 1270,1293,1377,1487, 1521, 1540,1615,1835,1935, 1983]
        true_drift_points = []
        true_abrupt_points = []
        true_incre_points = []
        outlier_list = []
    elif dataset_name == "micro":
        window_size = 1000
        eta = 1 / 8
        judge_size = 30  # 漂移前后30个点为有效范围
        rate = [0.1]
        n1 = 30
        n2 = 30

        data = pd.read_excel('./Dataset/micro.xlsx')
        data.replace('None', np.nan, inplace=True)

        # Remove rows containing NaN values
        data.dropna(subset=['input_voltage', 'el_power'], inplace=True)

        # Convert data to numeric type
        data['input_voltage'] = pd.to_numeric(data['input_voltage'], errors='coerce')
        data['el_power'] = pd.to_numeric(data['el_power'], errors='coerce')

        # Remove NaN values introduced by conversion
        data.dropna(subset=['input_voltage', 'el_power'], inplace=True)

        # Define Y and X
        X = data['el_power']
        Y = data['input_voltage']
        X = X.values.reshape(-1, 1)
        all_Y = Y.values.reshape(-1, 1)
        X = X.astype(np.float64)
        all_Y = all_Y.astype(np.float64)

        n1_kswin = int(len(all_Y) / 10)
        n2_kswin = int(len(all_Y) / 100)
        true_drift_points = [2797, 5004, 7625]
        true_abrupt_points = []
        true_incre_points = [2797, 5004, 7625]
        outlier_list = []

    n1 = 30
    # 定义每个检测器的参数网格
    param_grids = {
        "ADWIN": {"delta": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
                  "mint_min_window_longitude" :{'Type': 'integer', 'Mapping': list(range(1, n1))},
                    "mdbl_delta" : {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
                    "mint_clock" :{'Type': 'integer', 'Mapping': list(range(1, n1))},
                    "mint_min_window_length" : {'Type': 'integer', 'Mapping': list(range(1, n1))}},
        "DDM": {
            "min_num_instances": {'Type': 'integer', 'Mapping': list(range(1, n1))},
            "warning_level": {'Type': 'continuous', 'Range': [0, 2], 'Wrapper': lambda x: x},
            "out_control_level": {'Type': 'continuous', 'Range': [2, 4], 'Wrapper': lambda x: x}
        },
        "HDDM_W": {
            "drift_confidence": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
            "lambda_option": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
        },
        "KSWIN": {
            "alpha": {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
            "window_size": {'Type': 'integer', 'Mapping': list(range(n2_kswin+2, n1_kswin))},
            "stat_size": {'Type': 'integer', 'Mapping': list(range(1, n2_kswin))}
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
            "threshold": {'Type': 'continuous', 'Range': [0, 20], 'Wrapper': lambda x: x},
            "burn_in": {'Type': 'integer', 'Mapping': list(range(1, n1))},
            "k": {'Type': 'integer', 'Mapping': list(range(1, n1))}
            # "direction": ["positive"]
        },
        "EWMAD_DT": {
            "delta": {'Type': 'continuous', 'Range': [0, 0.1], 'Wrapper': lambda x: x},
            "threshold": {'Type': 'continuous', 'Range': [0, 0.5], 'Wrapper': lambda x: x},
            "burn_in": {'Type': 'integer', 'Mapping': list(range(1, n1))},
            "alpha": {'Type': 'continuous', 'Range': [0, 0.2], 'Wrapper': lambda x: x},
            "k": {'Type': 'integer', 'Mapping': list(range(5, 10))},
            "kk": {'Type': 'integer', 'Mapping': list(range(1, 20))},
        }
    }

    detectors = {
        "EWMAD_DT": EWMAD_DT(delta=0.001, threshold=0.8, burn_in=20, alpha=0.1, k=20, kk=20, direction="positive"),
        "ADWIN": ADWIN(delta=0.002),
        "KSWIN": KSWIN(alpha=0.000001, window_size=100, stat_size=30, data=None),
        "PageHinkley": PageHinkley(delta=0.001, threshold=5, burn_in=20, k=10, direction="positive"),
        "DataStream_Adapt": DataStream_Adapt(delta=0.001, gamma_thres=0.15, threshold=5, burn_in=20, k=10,
                                             direction="positive")
    }


    F1score = np.ones((len(rate), len(detectors), 8)) * (-1)
    results_0 = {
        "detector": None,          # 检测器名称（字符串）
        "Model": None,             # 模型名称（字符串）
        "TP": 0,                   # True Positives（整数）
        "FP": 0,                   # False Positives（整数）
        "FN": 0,                   # False Negatives（整数）
        "Precision": 0.0,          # 精确率（浮点数）
        "Recall": 0.0,             # 召回率（浮点数）
        "F1_Score_alldrift": 0.0,  # 漂移检测的 F1 分数（浮点数）
        "F1_Score_outlier": 0.0,   # 异常检测的 F1 分数（浮点数）
        "time": 0.0,               # 执行时间（浮点数，单位：秒）
        "mean_delay": float('inf'),  # 平均延迟（浮点数，初始设为无穷大）
        "mean_delay_abrupt": float('inf'),
        "mean_delay_incre": float('inf')
    }

    for num_r,r in enumerate(rate):
        # Define results path and create directory.
        path = './Results/'
        path += dataset_name + '_'
        path += "%s" % r + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        if r!=0:
            outlier_list = []
            for outlier in range(int(len(all_Y) * r)):
                # Generate mean and standard deviation for outliers
                mean = np.random.uniform(low=0.3*np.mean(all_Y), high=0.5*np.mean(all_Y), size=1)
                std_dev = np.random.uniform(low=0, high=1, size=1)
                random_number = np.random.normal(mean, std_dev, size=1)

                # 50% probability of generating a negative mean outlier
                if np.random.rand() < 0.5:
                    random_number = np.random.normal((-mean), std_dev, size=1)

                # Add outliers to a specific position (e.g., based on outlier index)
                outlier_index = 1 / r * (outlier + 1)
                if outlier_index < len(all_Y):  # Ensure the position exists
                    all_Y[int(outlier_index)] += random_number[0]
                    # print(f"Added outlier at index {outlier_index}: {random_number[0]}")
                outlier_list.append(outlier_index)
        # print(outlier_list)
        results_list = []
        # time_list = []
        # best_params = {}
        # best_scores = {}
        input_dim = X.shape[1]
        print(input_dim)
        for num_detector,(name_detector, detector) in enumerate(detectors.items()):
            print(name_detector)
            # 模型字典
            models = {
                "ARLF": FastAdaptiveRobustRegressor(input_dim=input_dim, hidden_dim=64, learning_rate=0.005),
                "$\Theta$-IPOD": IPCD(X=X, Y=all_Y, window_size=window_size, detector=detector, detetor_name=name_detector,eta=eta),
                "RANSACRegressor":RANSACRegressor(estimator=LinearRegression(), min_samples=2,max_trials=100,stop_probability=0.99,random_state=42),
                "HuberRegressor":HuberRegressor(epsilon=1.35,alpha=0.0001,max_iter=3000,tol=1e-5),
                "TheilSenRegressor":TheilSenRegressor(max_subpopulation=1e3, random_state=42)
            }
            # 循环遍历每个模型
            for num,(name, model) in enumerate(models.items()):
                print(f"Processing model: {name}")
                ParaSpace = param_grids[name_detector]
                clf = SeqUD(ParaSpace,n_runs_per_stage=20, max_runs=100, random_state=1, verbose=True)
                optimizer = OptimizationWrapper(
                    name_detector, true_drift_points, judge_size,
                    model, name, X, all_Y, window_size, eta
                )
                clf.fmax(optimizer)


                results_list.append(optimizer.best_results)
                print(num_detector,num_r,num_detector,num,optimizer.best_results,optimizer.best_drift_points,optimizer.best_incre,optimizer.best_incre_end,optimizer.best_outlier_points)
                F1score[num_r,num_detector,num] = optimizer.best_score


                if name_detector=="EWMAD_DT" or name_detector=="DataStream_Adapt" :
                    plt.figure(figsize=(12, 6))
                    plt.plot(X,color="purple", label="Humidity")
                    plt.plot(all_Y,color="black", label="Temperature")
                    for dp in optimizer.best_abrupt:
                        plt.axvline(dp, color="red", linestyle="--", label="Detected Abrupt Drift" if dp == optimizer.best_abrupt[0] else "")
                    for dp in optimizer.best_incre:
                        plt.axvline(dp, color="pink", linestyle="--", label="Detected Incremental Drift" if dp == optimizer.best_incre[0] else "")
                    # for dp in best_incre_end:
                    #     plt.axvline(dp, color="orange", linestyle="--", label="Detected Incremental end" if dp == best_incre_end[0] else "")
                    for op in optimizer.best_outlier_points:
                        plt.axvline(op, color="green", linestyle="--", label="Detected Outlier" if op == optimizer.best_outlier_points[0] else "")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.title(f"{name} and {name_detector} Drift Detection")
                    if name == "$\Theta$-IPOD":
                        plt.savefig(f"{path}/IPOD_and_{name_detector}_Drift_Detection{dataset_name}.png",
                                    bbox_inches="tight")
                    else:
                        plt.savefig(f"{path}/{name}_and_{name_detector}_Drift_Detection{dataset_name}.png",
                                    bbox_inches="tight")
                    # plt.show()
                else:
                    plt.figure(figsize=(12, 6))
                    plt.plot(X, color="purple", label="Humidity")
                    plt.plot(all_Y, color="black", label="Temperature")
                    for dp in optimizer.best_drift_points:
                        plt.axvline(dp, color="orange", linestyle="--",
                                    label="Detected Drift" if dp == optimizer.best_drift_points[0] else "")
                    for op in optimizer.best_outlier_points:
                        plt.axvline(op, color="green", linestyle="--",
                                    label="Detected Outlier" if op == optimizer.best_outlier_points[0] else "")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.title(f"{name} and {name_detector} Drift Detection")
                    if name == "$\Theta$-IPOD":
                        plt.savefig(f"{path}/IPOD_and_{name_detector}_Drift_Detection{dataset_name}.png", bbox_inches="tight")
                    else:
                        plt.savefig(f"{path}/{name}_and_{name_detector}_Drift_Detection{dataset_name}.png", bbox_inches="tight")
                    # plt.show()

        df_results = pd.DataFrame(results_list)
        # 创建全局参数表格
        global_params = {
            "Parameter": ["window_size", "judge_size", "X_dim", "rate","eta"],
            "Value": [window_size, judge_size, X.shape[1], r, eta]
        }
        df_global_params = pd.DataFrame(global_params)

        # Save to Excel
        with pd.ExcelWriter(f"%s/{r:.3f}out_{dataset_name}_drift_detection_results_summary_rate.xlsx" % path) as writer:
            df_global_params.to_excel(writer, sheet_name="Global Parameters", index=False)
            df_results.to_excel(writer, sheet_name="Model Results", index=False)

        print(f"Results summary saved to %s/{r:.3f}out_{dataset_name}_drift_detection_results_summary_rate.xlsx" % path)

