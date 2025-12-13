import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# =============================================================================
# 优化版模型 1: Fast Adaptive Robust Regressor
# 改进点: 加入 Mini-batch 训练
# =============================================================================

class FastAdaptiveRobustRegressor:
    def __init__(self, input_dim=1, epochs=20, learning_rate=0.005, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.epochs = epochs
        self.model = nn.Linear(input_dim, 1).to(device)

        # 自适应参数
        self.alpha = nn.Parameter(torch.tensor(1.0).to(device))
        self.scale = nn.Parameter(torch.tensor(1.0).to(device))

        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.alpha, self.scale],
            lr=learning_rate
        )

    def _barron_loss_function(self, residuals):
        # 限制参数范围以保证数值稳定
        alpha = self.alpha
        c = torch.sigmoid(self.scale) * 5.0 + 0.1

        x = residuals / c
        squared_term = (x ** 2)
        abs_alpha_minus_2 = torch.abs(alpha - 2.0) + 1e-6

        inner = squared_term / abs_alpha_minus_2 + 1.0
        loss_val = (abs_alpha_minus_2 / (alpha + 1e-6)) * (torch.pow(inner, alpha / 2.0) - 1.0)
        return torch.mean(loss_val)

    def fit(self, X, y, batch_size=64, verbose=False):
        """
        增加了 batch_size 参数，使用 DataLoader 加速
        """
        # 转换为 Tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        # 创建 DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()

                y_pred = self.model(batch_X)
                residuals = y_pred - batch_y
                loss = self._barron_loss_function(residuals)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"[Adaptive] Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}, Alpha={self.alpha.item():.2f}")

    def predict(self, X):
        self.model.eval()
        # 预测时如果数据量太大，也建议切分，这里简单处理
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).cpu().numpy()


# =============================================================================
# 优化版模型 2: Fast TTL + Stochastic HOVR
# 改进点:
# 1. Mini-batch 训练
# 2. Stochastic HOVR: 只在随机采样的子集上计算昂贵的梯度正则化
# =============================================================================

class Fast_TTL_HOVR_Regressor:
    def __init__(self, input_dim=1, epochs=20, learning_rate=0.005,
                 trim_ratio=0.2, reg_lambda=0.1, hovr_sample_rate=0.1, device='cpu'):
        """
        hovr_sample_rate: float (0.0 ~ 1.0).
            在计算输入梯度正则化时，只对 Batch 中 20% 的样本计算。
            这能极大地提速，因为 create_graph=True 非常慢。
        """
        self.device = device
        self.trim_ratio = trim_ratio
        self.reg_lambda = reg_lambda
        self.hovr_sample_rate = hovr_sample_rate
        self.epochs = epochs
        self.model = nn.Linear(input_dim, 1).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y, batch_size=64, verbose=False):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()

                # --- 1. TTL (Transformed Trimmed Loss) ---
                # 这一步不需要 requires_grad，正常跑
                y_pred = self.model(batch_X)
                raw_losses = (y_pred - batch_y) ** 2

                curr_batch_size = batch_X.shape[0]
                keep_k = int(curr_batch_size * (1 - self.trim_ratio))
                keep_k = max(keep_k, 1)

                sorted_losses, _ = torch.sort(raw_losses, dim=0)
                trimmed_loss = torch.mean(sorted_losses[:keep_k])

                # --- 2. Stochastic HOVR (修复后的逻辑) ---
                if self.reg_lambda > 0:
                    # A. 随机采样索引
                    n_sample = int(curr_batch_size * self.hovr_sample_rate)
                    n_sample = max(n_sample, 2)  # 至少保证有2个样本

                    perm = torch.randperm(curr_batch_size)
                    idx = perm[:n_sample]

                    # B. 【关键修复】:
                    # 1. 把采样出的 X 拿出来，detach 掉旧图，并开启梯度追踪
                    sampled_X = batch_X[idx].clone().detach()
                    sampled_X.requires_grad_(True)

                    # 2. 【关键修复】:
                    # 必须用 sampled_X 重新跑一次模型，建立新的计算图
                    # 这样 sampled_pred_hovr 才是直接由 sampled_X 算出来的
                    sampled_pred_hovr = self.model(sampled_X)

                    grad_outputs = torch.ones_like(sampled_pred_hovr)

                    # C. 现在求导就不会报错了，因为图是连通的
                    gradients = torch.autograd.grad(
                        outputs=sampled_pred_hovr,
                        inputs=sampled_X,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]

                    grad_norm = torch.sum(gradients ** 2, dim=1).mean()

                    total_loss = trimmed_loss + self.reg_lambda * grad_norm
                else:
                    total_loss = trimmed_loss

                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                # 避免除以0
                steps = len(dataloader)
                print(f"[Fast TTL] Epoch {epoch + 1}: Avg Loss={(epoch_loss / steps) if steps > 0 else 0:.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).cpu().numpy()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 类 1: R2T_Core
# 功能: 定义神经网络结构 (Transformer Encoder + Parameter Prediction Head)
# ==============================================================================
class R2T_Core(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, output_dim=2):
        """
        R2T 核心网络结构
        Args:
            input_dim (int): 输入特征 X 的维度 (通常为1)
            d_model (int): Transformer 的隐藏层维度
            nhead (int): 多头注意力的头数
            num_layers (int): Transformer 编码器层数
            output_dim (int): 需要预测的参数数量 (对于一元线性回归 y=wx+b，output_dim=2)
        """
        super(R2T_Core, self).__init__()

        # 1. 输入嵌入层: 将 (x, y) 拼接后映射到高维空间
        # 输入维度是 input_dim (x的特征) + 1 (y值)
        self.embedding = nn.Linear(input_dim + 1, d_model)

        # 2. 位置编码 (可选): 对于集合回归任务，顺序其实不重要，但为了保持Transformer结构完整性通常保留
        # 这里简化处理，直接依靠 Self-Attention 处理点与点之间的关系

        # 3. Transformer 编码器: 核心组件，用于捕捉全局上下文并识别异常值
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 注意力池化层 (Attention Pooling):
        # 并不是简单的取平均，而是让网络学习哪些点重要（Inliers），哪些点该忽略（Outliers）
        self.attention_scorer = nn.Linear(d_model, 1)

        # 5. 参数预测头 (Parameter Head): 将聚合后的特征映射为回归参数 (w, b)
        self.param_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # 输出 [w, b]
        )

    def forward(self, x, y):
        """
        Args:
            x: [Batch, Seq_Len, Input_Dim]
            y: [Batch, Seq_Len, 1]
        Returns:
            pred_params: [Batch, Output_Dim] (例如 [w, b])
            attention_weights: [Batch, Seq_Len, 1] (用于可视化模型关注了哪些点)
        """
        # 拼接 x 和 y，形成点集序列 [(x1,y1), (x2,y2), ...]
        # shape: [Batch, Seq_Len, Input_Dim + 1]
        combined_input = torch.cat([x, y], dim=-1)

        # 映射到 latent space
        features = self.embedding(combined_input)  # [Batch, Seq_Len, d_model]

        # Transformer 编码
        encoded_features = self.transformer_encoder(features)  # [Batch, Seq_Len, d_model]

        # 计算注意力分数 (Attention Pooling)
        # 形状: [Batch, Seq_Len, 1]
        attn_logits = self.attention_scorer(encoded_features)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # 加权聚合特征 (Weighted Sum)
        # 异常值的 attn_weights 应该趋近于 0
        pooled_features = torch.sum(encoded_features * attn_weights, dim=1)  # [Batch, d_model]

        # 预测回归参数
        pred_params = self.param_predictor(pooled_features)

        return pred_params, attn_weights


class R2T_Regressor:
    def __init__(self, input_dim=1, learning_rate=0.005, epochs=30, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.epochs = epochs

        # 初始化核心网络
        self.model = R2T_Core(input_dim=input_dim, output_dim=input_dim + 1).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # 用于保存训练好的回归参数 (w, b)
        self.learned_w = None
        self.learned_b = None

    def fit(self, X, y):
        """
        训练函数：不仅优化网络，还要保存最终学到的 w 和 b
        """
        # 确保 y 是 (N, 1) 形状
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # [1, N, Dim]
        y_tensor = torch.FloatTensor(y).unsqueeze(0).to(self.device)  # [1, N, 1]

        self.model.train()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # 1. 预测参数
            pred_params, _ = self.model(X_tensor, y_tensor)
            w_pred = pred_params[0, :-1]
            b_pred = pred_params[0, -1]

            # 2. 重构 Y
            y_reconstructed = torch.matmul(X_tensor, w_pred.unsqueeze(1)) + b_pred

            # 3. 计算 Loss 并更新
            loss = self.criterion(y_reconstructed, y_tensor)
            loss.backward()
            self.optimizer.step()

        # ====================================================
        # 【关键修复】：训练结束后，保存学到的参数用于预测
        # ====================================================
        self.model.eval()
        with torch.no_grad():
            # 再跑一次前向传播，获取最终稳定的参数
            final_params, _ = self.model(X_tensor, y_tensor)
            self.learned_w = final_params[0, :-1].cpu()  # 转回 CPU 以便后续计算
            self.learned_b = final_params[0, -1].cpu()

        # print(f"R2T Training Finished. Learned Params: w={self.learned_w}, b={self.learned_b}")

    def predict(self, X):
        """
        预测函数：使用 fit 阶段学到的参数进行线性回归计算
        """
        if self.learned_w is None or self.learned_b is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")

        self.model.eval()

        # 1. 数据转换
        # 确保输入是 Tensor，形状匹配
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X.float()

        # 如果 X 是 [N, 1]，我们直接计算 y = Xw + b
        # 注意 self.learned_w 的形状可能是 [1] 或 [Dim]

        with torch.no_grad():
            # 简单的线性回归公式: y = X * w + b
            # X_tensor: [N, input_dim]
            # learned_w: [input_dim] -> unsqueeze -> [input_dim, 1]
            y_pred = torch.matmul(X_tensor, self.learned_w.unsqueeze(1)) + self.learned_b

        # 2. 返回 Numpy 数组 (这里必须返回数组，因为你的报错显示正在做减法运算)
        return y_pred.numpy().flatten()  # 或者 .ravel()
