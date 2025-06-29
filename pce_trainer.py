#!/usr/bin/env python3
"""
PCE (Polynomial Chaos Expansion) Trainer with Intelligent Order Selection
用于训练PCE模型替代神经网络的Python程序

PCE使用多项式基函数来近似复杂的输入-输出关系
集成了智能阶数选择功能，可以自动分析函数非线性强度并推荐最优阶数
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.fft import fft, fftfreq
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class PCETrainer:
    def __init__(self, input_dim=2, output_dim=78, polynomial_order=None, auto_order_selection=True):
        """
        初始化PCE训练器

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            polynomial_order: 多项式阶数 (None表示自动选择)
            auto_order_selection: 是否启用自动阶数选择
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.polynomial_order = polynomial_order
        self.auto_order_selection = auto_order_selection

        # 如果没有指定阶数且启用自动选择，则设为None等待后续确定
        if self.polynomial_order is None and not auto_order_selection:
            self.polynomial_order = 2  # 默认2阶

        # 计算多项式基函数的数量（如果阶数已确定）
        if self.polynomial_order is not None:
            self.n_basis = self._calculate_basis_count()
        else:
            self.n_basis = None

        # PCE系数矩阵 (output_dim x n_basis)
        self.coefficients = None

        # 数据标准化器
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # 阶数选择相关
        self.order_selection_results = None

        print(f"PCE Trainer initialized:")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Output dimension: {self.output_dim}")
        if self.polynomial_order is not None:
            print(f"  Polynomial order: {self.polynomial_order}")
            print(f"  Number of basis functions: {self.n_basis}")
        else:
            print(f"  Polynomial order: Auto-selection enabled")
        print(f"  Auto order selection: {self.auto_order_selection}")
    
    def _calculate_basis_count(self):
        """计算多项式基函数数量"""
        if self.polynomial_order is None:
            return None

        # 通用公式: C(n+d, d) where n=input_dim, d=polynomial_order
        from math import comb
        return comb(self.input_dim + self.polynomial_order, self.polynomial_order)

    def analyze_nonlinearity(self, X, Y):
        """分析函数非线性强度"""
        print("🔍 分析函数非线性强度...")

        # 标准化数据
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1) if Y.ndim == 1 else Y)

        metrics = {}

        # 1. 线性相关性分析
        metrics['linear_correlation'] = self._analyze_linear_correlation(X_scaled, Y_scaled)

        # 2. 高阶矩分析
        metrics['higher_moments'] = self._analyze_higher_moments(Y_scaled)

        # 3. 频域分析
        metrics['frequency_analysis'] = self._analyze_frequency_domain(Y_scaled)

        # 4. 局部线性度分析
        metrics['local_linearity'] = self._analyze_local_linearity(X_scaled, Y_scaled)

        # 5. 梯度变化分析
        metrics['gradient_variation'] = self._analyze_gradient_variation(X_scaled, Y_scaled)

        return metrics

    def _analyze_linear_correlation(self, X_scaled, Y_scaled):
        """线性相关性分析"""
        n_outputs = Y_scaled.shape[1] if Y_scaled.ndim > 1 else 1
        correlations = []

        for i in range(n_outputs):
            y = Y_scaled[:, i] if Y_scaled.ndim > 1 else Y_scaled

            # 计算与每个输入的线性相关性
            corr_with_inputs = []
            for j in range(self.input_dim):
                corr = np.corrcoef(X_scaled[:, j], y)[0, 1]
                if not np.isnan(corr):
                    corr_with_inputs.append(abs(corr))

            # 计算与输入乘积项的相关性
            interaction_corrs = []
            for j in range(self.input_dim):
                for k in range(j+1, self.input_dim):
                    interaction = X_scaled[:, j] * X_scaled[:, k]
                    corr = np.corrcoef(interaction, y)[0, 1]
                    if not np.isnan(corr):
                        interaction_corrs.append(abs(corr))

            # 计算与平方项的相关性
            quadratic_corrs = []
            for j in range(self.input_dim):
                quadratic = X_scaled[:, j] ** 2
                corr = np.corrcoef(quadratic, y)[0, 1]
                if not np.isnan(corr):
                    quadratic_corrs.append(abs(corr))

            correlations.append({
                'linear': np.mean(corr_with_inputs) if corr_with_inputs else 0,
                'interaction': np.mean(interaction_corrs) if interaction_corrs else 0,
                'quadratic': np.mean(quadratic_corrs) if quadratic_corrs else 0
            })

        avg_linear = np.mean([c['linear'] for c in correlations])
        avg_interaction = np.mean([c['interaction'] for c in correlations])
        avg_quadratic = np.mean([c['quadratic'] for c in correlations])

        return {
            'avg_linear_corr': avg_linear,
            'avg_interaction_corr': avg_interaction,
            'avg_quadratic_corr': avg_quadratic,
            'nonlinearity_ratio': (avg_quadratic + avg_interaction) / (avg_linear + 1e-8)
        }

    def _analyze_higher_moments(self, Y_scaled):
        """高阶矩分析"""
        n_outputs = Y_scaled.shape[1] if Y_scaled.ndim > 1 else 1
        skewness_values = []
        kurtosis_values = []

        for i in range(n_outputs):
            y = Y_scaled[:, i] if Y_scaled.ndim > 1 else Y_scaled
            skewness = abs(stats.skew(y))
            kurtosis = abs(stats.kurtosis(y))
            skewness_values.append(skewness)
            kurtosis_values.append(kurtosis)

        return {
            'avg_skewness': np.mean(skewness_values),
            'avg_kurtosis': np.mean(kurtosis_values),
            'moment_complexity': np.mean(skewness_values) + np.mean(kurtosis_values)
        }

    def _analyze_frequency_domain(self, Y_scaled):
        """频域分析"""
        n_outputs = Y_scaled.shape[1] if Y_scaled.ndim > 1 else 1
        high_freq_ratios = []

        for i in range(n_outputs):
            y = Y_scaled[:, i] if Y_scaled.ndim > 1 else Y_scaled

            # FFT分析
            fft_y = fft(y)
            freqs = fftfreq(len(y))

            # 计算高频成分比例
            power_spectrum = np.abs(fft_y) ** 2
            total_power = np.sum(power_spectrum)

            # 定义高频为频率 > 0.1 (归一化频率)
            high_freq_mask = np.abs(freqs) > 0.1
            high_freq_power = np.sum(power_spectrum[high_freq_mask])

            high_freq_ratio = high_freq_power / (total_power + 1e-8)
            high_freq_ratios.append(high_freq_ratio)

        return {
            'avg_high_freq_ratio': np.mean(high_freq_ratios),
            'max_high_freq_ratio': np.max(high_freq_ratios)
        }

    def _analyze_local_linearity(self, X_scaled, Y_scaled):
        """局部线性度分析"""
        n_samples = X_scaled.shape[0]
        k = min(10, n_samples // 4)  # 选择合适的k值

        try:
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
            local_linearities = []

            # 采样分析
            sample_indices = np.random.choice(n_samples, min(50, n_samples), replace=False)

            for i in sample_indices:
                # 找到k个最近邻
                distances, indices = nbrs.kneighbors([X_scaled[i]])

                # 在局部区域拟合线性模型
                local_X = X_scaled[indices[0]]
                local_Y = Y_scaled[indices[0]] if Y_scaled.ndim > 1 else Y_scaled[indices[0]]

                if local_Y.ndim == 1:
                    local_Y = local_Y.reshape(-1, 1)

                local_r2_scores = []
                for j in range(local_Y.shape[1]):
                    try:
                        lr = LinearRegression()
                        lr.fit(local_X, local_Y[:, j])
                        local_pred = lr.predict(local_X)
                        local_r2 = r2_score(local_Y[:, j], local_pred)
                        local_r2_scores.append(max(0, local_r2))
                    except:
                        local_r2_scores.append(0)

                local_linearities.append(np.mean(local_r2_scores))

            return {
                'avg_local_linearity': np.mean(local_linearities),
                'min_local_linearity': np.min(local_linearities)
            }
        except:
            return {
                'avg_local_linearity': 0.5,
                'min_local_linearity': 0.5
            }

    def _analyze_gradient_variation(self, X_scaled, Y_scaled):
        """梯度变化分析"""
        n_outputs = Y_scaled.shape[1] if Y_scaled.ndim > 1 else 1
        gradients = []

        for i in range(n_outputs):
            y = Y_scaled[:, i] if Y_scaled.ndim > 1 else Y_scaled

            grad_variations = []
            for j in range(self.input_dim):
                try:
                    # 排序以计算梯度
                    sorted_indices = np.argsort(X_scaled[:, j])
                    sorted_x = X_scaled[sorted_indices, j]
                    sorted_y = y[sorted_indices]

                    # 计算数值梯度
                    grad = np.gradient(sorted_y, sorted_x)

                    # 计算梯度的变化程度
                    grad_variation = np.std(grad) / (np.mean(np.abs(grad)) + 1e-8)
                    grad_variations.append(grad_variation)
                except:
                    grad_variations.append(1.0)

            gradients.append(np.mean(grad_variations))

        return {
            'avg_gradient_variation': np.mean(gradients),
            'max_gradient_variation': np.max(gradients)
        }

    def select_optimal_order(self, X, Y, max_order=5):
        """智能选择最优PCE阶数"""
        print("🎯 智能PCE阶数选择")
        print("=" * 50)

        # 1. 非线性强度分析
        nonlinearity_metrics = self.analyze_nonlinearity(X, Y)

        # 2. 基于理论的阶数建议
        theory_order = self._theory_based_order_selection(nonlinearity_metrics)

        # 3. 交叉验证选择
        cv_order = self._cross_validation_order_selection(X, Y, max_order)

        # 4. 信息准则选择
        ic_order = self._information_criterion_selection(X, Y, max_order)

        # 5. 综合决策
        optimal_order = self._make_final_decision(theory_order, cv_order, ic_order)

        # 保存选择结果
        self.order_selection_results = {
            'optimal_order': optimal_order,
            'nonlinearity_metrics': nonlinearity_metrics,
            'theory_order': theory_order,
            'cv_order': cv_order,
            'ic_order': ic_order
        }

        # 更新PCE配置
        self.polynomial_order = optimal_order
        self.n_basis = self._calculate_basis_count()

        print(f"✅ 最优阶数选择完成: {optimal_order}")
        print(f"   基函数数量: {self.n_basis}")

        return optimal_order

    def _theory_based_order_selection(self, metrics):
        """基于理论的阶数选择"""
        print("📊 基于非线性强度理论分析...")

        # 计算综合非线性强度分数
        linear_corr = metrics['linear_correlation']['avg_linear_corr']
        nonlin_ratio = metrics['linear_correlation']['nonlinearity_ratio']
        moment_complexity = metrics['higher_moments']['moment_complexity']
        high_freq_ratio = metrics['frequency_analysis']['avg_high_freq_ratio']
        local_linearity = metrics['local_linearity']['avg_local_linearity']
        grad_variation = metrics['gradient_variation']['avg_gradient_variation']

        # 归一化并计算综合分数
        nonlinearity_score = (
            (1 - linear_corr) * 0.2 +
            min(nonlin_ratio, 2.0) * 0.2 +
            min(moment_complexity, 5.0) * 0.15 +
            min(high_freq_ratio, 1.0) * 0.2 +
            (1 - local_linearity) * 0.15 +
            min(grad_variation, 3.0) * 0.1
        )

        # 基于分数建议阶数
        if nonlinearity_score < 0.3:
            suggested_order = 1
            reason = "函数接近线性"
        elif nonlinearity_score < 0.6:
            suggested_order = 2
            reason = "中等非线性强度"
        elif nonlinearity_score < 1.0:
            suggested_order = 3
            reason = "较强非线性"
        elif nonlinearity_score < 1.5:
            suggested_order = 4
            reason = "强非线性"
        else:
            suggested_order = 5
            reason = "极强非线性"

        print(f"   非线性强度分数: {nonlinearity_score:.3f}")
        print(f"   理论建议阶数: {suggested_order} ({reason})")

        return {
            'suggested_order': suggested_order,
            'nonlinearity_score': nonlinearity_score,
            'reason': reason
        }

    def _cross_validation_order_selection(self, X, Y, max_order):
        """交叉验证选择阶数"""
        print("🔄 交叉验证测试不同阶数...")

        orders = range(1, min(max_order + 1, 6))
        cv_scores = {}

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for order in orders:
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                Y_train, Y_val = Y[train_idx], Y[val_idx]

                try:
                    score = self._fit_evaluate_pce_order(X_train, Y_train, X_val, Y_val, order)
                    scores.append(score)
                except Exception as e:
                    print(f"   警告: 阶数{order}测试失败: {e}")
                    scores.append(0)

            cv_scores[order] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }

            print(f"   阶数{order}: R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

        best_order = max(cv_scores.keys(), key=lambda k: cv_scores[k]['mean_score'])
        print(f"   交叉验证最佳阶数: {best_order}")

        return {
            'best_order': best_order,
            'cv_scores': cv_scores
        }

    def _fit_evaluate_pce_order(self, X_train, Y_train, X_val, Y_val, order):
        """拟合和评估指定阶数的PCE模型"""
        # 创建多项式特征
        poly = PolynomialFeatures(degree=order, include_bias=True)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        # 训练模型
        model = Ridge(alpha=1e-6)

        if Y_train.ndim == 1:
            model.fit(X_train_poly, Y_train)
            Y_pred = model.predict(X_val_poly)
            return max(0, r2_score(Y_val, Y_pred))
        else:
            scores = []
            for i in range(Y_train.shape[1]):
                model.fit(X_train_poly, Y_train[:, i])
                Y_pred = model.predict(X_val_poly)
                score = r2_score(Y_val[:, i], Y_pred)
                scores.append(max(0, score))
            return np.mean(scores)

    def _information_criterion_selection(self, X, Y, max_order):
        """信息准则选择"""
        print("📈 信息准则分析...")

        orders = range(1, min(max_order + 1, 6))
        ic_results = {}

        for order in orders:
            try:
                aic, bic = self._compute_information_criteria(X, Y, order)
                ic_results[order] = {'AIC': aic, 'BIC': bic}
                print(f"   阶数{order}: AIC = {aic:.2f}, BIC = {bic:.2f}")
            except Exception as e:
                print(f"   警告: 阶数{order}信息准则计算失败: {e}")
                ic_results[order] = {'AIC': float('inf'), 'BIC': float('inf')}

        best_aic_order = min(ic_results.keys(), key=lambda k: ic_results[k]['AIC'])
        best_bic_order = min(ic_results.keys(), key=lambda k: ic_results[k]['BIC'])

        print(f"   AIC最佳阶数: {best_aic_order}")
        print(f"   BIC最佳阶数: {best_bic_order}")

        return {
            'best_aic_order': best_aic_order,
            'best_bic_order': best_bic_order,
            'ic_results': ic_results
        }

    def _compute_information_criteria(self, X, Y, order):
        """计算AIC和BIC"""
        poly = PolynomialFeatures(degree=order, include_bias=True)
        X_poly = poly.fit_transform(X)

        n_samples, n_features = X_poly.shape
        model = LinearRegression()

        if Y.ndim == 1:
            model.fit(X_poly, Y)
            Y_pred = model.predict(X_poly)
            mse = mean_squared_error(Y, Y_pred)
        else:
            mses = []
            for i in range(Y.shape[1]):
                model.fit(X_poly, Y[:, i])
                Y_pred = model.predict(X_poly)
                mse = mean_squared_error(Y[:, i], Y_pred)
                mses.append(mse)
            mse = np.mean(mses)

        # 计算AIC和BIC
        log_likelihood = -n_samples * np.log(mse + 1e-8) / 2
        aic = 2 * n_features - 2 * log_likelihood
        bic = np.log(n_samples) * n_features - 2 * log_likelihood

        return aic, bic

    def _make_final_decision(self, theory_result, cv_result, ic_result):
        """综合决策"""
        print("🎯 综合决策...")

        suggestions = [
            theory_result['suggested_order'],
            cv_result['best_order'],
            ic_result['best_aic_order'],
            ic_result['best_bic_order']
        ]

        # 计算加权平均
        weights = [0.4, 0.3, 0.15, 0.15]
        weighted_order = sum(s * w for s, w in zip(suggestions, weights))
        final_order = max(1, int(round(weighted_order)))

        print(f"   理论建议: {theory_result['suggested_order']}")
        print(f"   交叉验证: {cv_result['best_order']}")
        print(f"   AIC建议: {ic_result['best_aic_order']}")
        print(f"   BIC建议: {ic_result['best_bic_order']}")
        print(f"   加权平均: {weighted_order:.2f}")
        print(f"   最终决策: {final_order}")

        return final_order
    
    def _compute_basis_functions(self, X):
        """
        计算多项式基函数（支持任意阶数）

        Args:
            X: 输入数据 (n_samples, input_dim)

        Returns:
            basis_matrix: 基函数矩阵 (n_samples, n_basis)
        """
        # 使用sklearn的PolynomialFeatures来生成基函数
        poly = PolynomialFeatures(degree=self.polynomial_order, include_bias=True)
        Phi = poly.fit_transform(X)

        # 确保基函数数量匹配
        if Phi.shape[1] != self.n_basis:
            print(f"⚠️  基函数数量调整: 期望{self.n_basis} → 实际{Phi.shape[1]}")
            self.n_basis = Phi.shape[1]

        return Phi
    
    def generate_training_data(self, n_samples=1000, noise_level=0.01):
        """
        生成训练数据（模拟一个复杂的非线性函数）
        
        Args:
            n_samples: 样本数量
            noise_level: 噪声水平
            
        Returns:
            X: 输入数据 (n_samples, input_dim)
            Y: 输出数据 (n_samples, output_dim)
        """
        print(f"Generating {n_samples} training samples...")
        
        # 生成随机输入数据 (在[-1, 1]范围内)
        X = np.random.uniform(-1, 1, (n_samples, self.input_dim))
        
        # 生成复杂的非线性输出
        Y = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            x1, x2 = X[i, 0], X[i, 1]
            
            # 为每个输出维度定义不同的非线性函数
            for j in range(self.output_dim):
                # 创建复杂的非线性关系
                base_func = (
                    0.5 * np.sin(2 * np.pi * x1 + j * 0.1) * np.cos(np.pi * x2) +
                    0.3 * (x1**2 + x2**2) * np.exp(-0.5 * (x1**2 + x2**2)) +
                    0.2 * x1 * x2 * np.sin(j * 0.05) +
                    0.1 * (x1**3 - x2**3) +
                    j * 0.01  # 添加输出维度相关的偏移
                )
                
                # 添加噪声
                Y[i, j] = base_func + np.random.normal(0, noise_level)
        
        print(f"Training data generated successfully!")
        print(f"  Input range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Output range: [{Y.min():.3f}, {Y.max():.3f}]")
        
        return X, Y
    
    def train(self, X, Y, test_size=0.2, regularization=1e-6, max_order=5):
        """
        训练PCE模型（集成智能阶数选择）

        Args:
            X: 输入数据
            Y: 输出数据
            test_size: 测试集比例
            regularization: 正则化参数
            max_order: 自动选择时的最大阶数
        """
        print(f"\n🚀 开始PCE模型训练 (样本数: {len(X)})")

        # 智能阶数选择
        if self.auto_order_selection and self.polynomial_order is None:
            print("\n📊 启动智能阶数选择...")
            self.select_optimal_order(X, Y, max_order)
        elif self.polynomial_order is None:
            print("⚠️  未指定阶数且未启用自动选择，使用默认阶数2")
            self.polynomial_order = 2
            self.n_basis = self._calculate_basis_count()

        print(f"\n✅ 使用PCE阶数: {self.polynomial_order}")
        print(f"   基函数数量: {self.n_basis}")

        # 检查样本数量充足性
        min_samples_required = self.n_basis * 5
        if len(X) < min_samples_required:
            print(f"⚠️  警告: 样本数量({len(X)})可能不足，建议至少{min_samples_required}个样本")

        # 分割训练和测试数据
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )

        print(f"\n📊 数据分割:")
        print(f"   训练样本: {len(X_train)}")
        print(f"   测试样本: {len(X_test)}")

        # 标准化数据
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_test_scaled = self.input_scaler.transform(X_test)
        Y_train_scaled = self.output_scaler.fit_transform(Y_train)
        Y_test_scaled = self.output_scaler.transform(Y_test)

        # 计算基函数矩阵
        print("\n🔧 计算多项式基函数...")
        Phi_train = self._compute_basis_functions(X_train_scaled)
        Phi_test = self._compute_basis_functions(X_test_scaled)
        
        # 使用最小二乘法求解PCE系数
        # 添加正则化以提高数值稳定性
        A = Phi_train.T @ Phi_train + regularization * np.eye(self.n_basis)
        
        self.coefficients = np.zeros((self.output_dim, self.n_basis))
        
        for i in range(self.output_dim):
            b = Phi_train.T @ Y_train_scaled[:, i]
            self.coefficients[i, :] = np.linalg.solve(A, b)
        
        # 评估模型性能
        Y_train_pred = self._predict_scaled(X_train_scaled)
        Y_test_pred = self._predict_scaled(X_test_scaled)
        
        # 反标准化预测结果
        Y_train_pred = self.output_scaler.inverse_transform(Y_train_pred)
        Y_test_pred = self.output_scaler.inverse_transform(Y_test_pred)
        
        # 计算误差指标
        train_mse = mean_squared_error(Y_train, Y_train_pred)
        test_mse = mean_squared_error(Y_test, Y_test_pred)
        train_r2 = r2_score(Y_train, Y_train_pred)
        test_r2 = r2_score(Y_test, Y_test_pred)
        
        print(f"Training completed!")
        print(f"  Training MSE: {train_mse:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Training R2: {train_r2:.6f}")
        print(f"  Test R2: {test_r2:.6f}")
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'X_test': X_test,
            'Y_test': Y_test,
            'Y_test_pred': Y_test_pred
        }
    
    def _predict_scaled(self, X_scaled):
        """使用标准化后的输入进行预测"""
        Phi = self._compute_basis_functions(X_scaled)
        return Phi @ self.coefficients.T
    
    def predict(self, X):
        """
        使用训练好的PCE模型进行预测
        
        Args:
            X: 输入数据
            
        Returns:
            Y_pred: 预测输出
        """
        if self.coefficients is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        X_scaled = self.input_scaler.transform(X)
        Y_pred_scaled = self._predict_scaled(X_scaled)
        return self.output_scaler.inverse_transform(Y_pred_scaled)
    
    def save_model(self, filename='pce_model.pkl'):
        """保存训练好的模型"""
        model_data = {
            'coefficients': self.coefficients,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'polynomial_order': self.polynomial_order,
            'n_basis': self.n_basis
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='pce_model.pkl'):
        """加载训练好的模型"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.coefficients = model_data['coefficients']
        self.input_scaler = model_data['input_scaler']
        self.output_scaler = model_data['output_scaler']
        self.input_dim = model_data['input_dim']
        self.output_dim = model_data['output_dim']
        self.polynomial_order = model_data['polynomial_order']
        self.n_basis = model_data['n_basis']
        
        print(f"Model loaded from {filename}")
    
    def export_fortran_coefficients(self, filename='pce_coefficients.txt'):
        """导出系数到Fortran格式的文件"""
        if self.coefficients is None:
            raise ValueError("模型尚未训练")
        
        with open(filename, 'w') as f:
            f.write("! PCE Coefficients for Fortran\n")
            f.write(f"! Input dimension: {self.input_dim}\n")
            f.write(f"! Output dimension: {self.output_dim}\n")
            f.write(f"! Polynomial order: {self.polynomial_order}\n")
            f.write(f"! Number of basis functions: {self.n_basis}\n")
            f.write("!\n")
            f.write("! Coefficients matrix (output_dim x n_basis)\n")
            f.write("real*8 coeff(78,6)\n")
            f.write("data coeff / &\n")
            
            for i in range(self.output_dim):
                line = ", ".join([f"{coeff:.8f}d0" for coeff in self.coefficients[i, :]])
                if i < self.output_dim - 1:
                    f.write(f"  {line}, &\n")
                else:
                    f.write(f"  {line}  &\n")
            f.write("  /\n")
        
        print(f"Fortran coefficients exported to {filename}")

def main():
    """主函数：演示PCE训练过程"""
    print("=" * 60)
    print("PCE Neural Network Replacement Training Demo")
    print("=" * 60)
    
    # 创建PCE训练器
    trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
    
    # 生成训练数据
    X, Y = trainer.generate_training_data(n_samples=2000, noise_level=0.01)
    
    # 训练模型
    results = trainer.train(X, Y, test_size=0.2, regularization=1e-6)
    
    # 保存模型
    trainer.save_model('pce_model.pkl')
    
    # 导出Fortran系数
    trainer.export_fortran_coefficients('pce_coefficients_new.txt')
    
    # 可视化结果（仅显示前几个输出维度）
    plt.figure(figsize=(15, 5))
    
    for i in range(min(3, trainer.output_dim)):
        plt.subplot(1, 3, i+1)
        plt.scatter(results['Y_test'][:, i], results['Y_test_pred'][:, i], alpha=0.6)
        plt.plot([results['Y_test'][:, i].min(), results['Y_test'][:, i].max()], 
                 [results['Y_test'][:, i].min(), results['Y_test'][:, i].max()], 'r--')
        plt.xlabel(f'True Output {i+1}')
        plt.ylabel(f'Predicted Output {i+1}')
        plt.title(f'Output {i+1} Prediction')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pce_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining completed successfully!")
    print("Files generated:")
    print("  - pce_model.pkl: 完整的Python模型")
    print("  - pce_coefficients_new.txt: Fortran系数文件")
    print("  - pce_training_results.png: 训练结果可视化")

if __name__ == "__main__":
    main()
