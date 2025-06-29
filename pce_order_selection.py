#!/usr/bin/env python3
"""
PCE最优阶数选择工具
通过分析函数非线性强度来决定PCE的最优多项式阶数

理论基础：
1. 非线性强度指标
2. 信息论准则 (AIC, BIC)
3. 交叉验证
4. 样本复杂度分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class NonlinearityAnalyzer:
    """非线性强度分析器"""
    
    def __init__(self, X, Y):
        """
        初始化分析器
        
        Args:
            X: 输入数据 (n_samples, n_features)
            Y: 输出数据 (n_samples, n_outputs)
        """
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_outputs = Y.shape[1] if Y.ndim > 1 else 1
        
        # 标准化数据
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.Y_scaled = self.scaler_Y.fit_transform(Y.reshape(-1, 1) if Y.ndim == 1 else Y)
        
    def compute_nonlinearity_metrics(self):
        """计算多种非线性强度指标"""
        metrics = {}
        
        # 1. 线性相关性分析
        metrics['linear_correlation'] = self._linear_correlation_analysis()
        
        # 2. 高阶矩分析
        metrics['higher_moments'] = self._higher_moments_analysis()
        
        # 3. 频域分析
        metrics['frequency_analysis'] = self._frequency_domain_analysis()
        
        # 4. 局部线性度分析
        metrics['local_linearity'] = self._local_linearity_analysis()
        
        # 5. 梯度变化分析
        metrics['gradient_variation'] = self._gradient_variation_analysis()
        
        return metrics
    
    def _linear_correlation_analysis(self):
        """线性相关性分析"""
        correlations = []
        
        for i in range(self.n_outputs):
            y = self.Y_scaled[:, i] if self.Y_scaled.ndim > 1 else self.Y_scaled
            
            # 计算与每个输入的线性相关性
            corr_with_inputs = []
            for j in range(self.n_features):
                corr = np.corrcoef(self.X_scaled[:, j], y)[0, 1]
                corr_with_inputs.append(abs(corr))
            
            # 计算与输入乘积项的相关性
            interaction_corrs = []
            for j in range(self.n_features):
                for k in range(j+1, self.n_features):
                    interaction = self.X_scaled[:, j] * self.X_scaled[:, k]
                    corr = np.corrcoef(interaction, y)[0, 1]
                    interaction_corrs.append(abs(corr))
            
            # 计算与平方项的相关性
            quadratic_corrs = []
            for j in range(self.n_features):
                quadratic = self.X_scaled[:, j] ** 2
                corr = np.corrcoef(quadratic, y)[0, 1]
                quadratic_corrs.append(abs(corr))
            
            correlations.append({
                'linear': np.mean(corr_with_inputs),
                'interaction': np.mean(interaction_corrs) if interaction_corrs else 0,
                'quadratic': np.mean(quadratic_corrs)
            })
        
        return {
            'avg_linear_corr': np.mean([c['linear'] for c in correlations]),
            'avg_interaction_corr': np.mean([c['interaction'] for c in correlations]),
            'avg_quadratic_corr': np.mean([c['quadratic'] for c in correlations]),
            'nonlinearity_ratio': np.mean([c['quadratic'] + c['interaction'] for c in correlations]) / 
                                 (np.mean([c['linear'] for c in correlations]) + 1e-8)
        }
    
    def _higher_moments_analysis(self):
        """高阶矩分析"""
        moments = {}
        
        for i in range(self.n_outputs):
            y = self.Y_scaled[:, i] if self.Y_scaled.ndim > 1 else self.Y_scaled
            
            # 计算偏度和峰度
            skewness = stats.skew(y)
            kurtosis = stats.kurtosis(y)
            
            moments[f'output_{i}'] = {
                'skewness': abs(skewness),
                'kurtosis': abs(kurtosis)
            }
        
        avg_skewness = np.mean([m['skewness'] for m in moments.values()])
        avg_kurtosis = np.mean([m['kurtosis'] for m in moments.values()])
        
        return {
            'avg_skewness': avg_skewness,
            'avg_kurtosis': avg_kurtosis,
            'moment_complexity': avg_skewness + avg_kurtosis
        }
    
    def _frequency_domain_analysis(self):
        """频域分析 - 检测高频成分"""
        high_freq_ratios = []
        
        for i in range(self.n_outputs):
            y = self.Y_scaled[:, i] if self.Y_scaled.ndim > 1 else self.Y_scaled
            
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
    
    def _local_linearity_analysis(self):
        """局部线性度分析"""
        # 使用k近邻分析局部线性度
        from sklearn.neighbors import NearestNeighbors
        
        k = min(10, self.n_samples // 4)  # 选择合适的k值
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.X_scaled)
        
        local_linearities = []
        
        for i in range(min(100, self.n_samples)):  # 采样分析
            # 找到k个最近邻
            distances, indices = nbrs.kneighbors([self.X_scaled[i]])
            
            # 在局部区域拟合线性模型
            local_X = self.X_scaled[indices[0]]
            local_Y = self.Y_scaled[indices[0]] if self.Y_scaled.ndim > 1 else self.Y_scaled[indices[0]]
            
            # 计算局部线性拟合的R²
            if local_Y.ndim == 1:
                local_Y = local_Y.reshape(-1, 1)
            
            local_r2_scores = []
            for j in range(local_Y.shape[1]):
                try:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(local_X, local_Y[:, j])
                    local_pred = lr.predict(local_X)
                    local_r2 = r2_score(local_Y[:, j], local_pred)
                    local_r2_scores.append(max(0, local_r2))  # 避免负值
                except:
                    local_r2_scores.append(0)
            
            local_linearities.append(np.mean(local_r2_scores))
        
        return {
            'avg_local_linearity': np.mean(local_linearities),
            'min_local_linearity': np.min(local_linearities)
        }
    
    def _gradient_variation_analysis(self):
        """梯度变化分析"""
        # 计算数值梯度的变化
        gradients = []
        
        for i in range(self.n_outputs):
            y = self.Y_scaled[:, i] if self.Y_scaled.ndim > 1 else self.Y_scaled
            
            # 计算每个维度的梯度
            grad_variations = []
            for j in range(self.n_features):
                # 排序以计算梯度
                sorted_indices = np.argsort(self.X_scaled[:, j])
                sorted_x = self.X_scaled[sorted_indices, j]
                sorted_y = y[sorted_indices]
                
                # 计算数值梯度
                grad = np.gradient(sorted_y, sorted_x)
                
                # 计算梯度的变化程度
                grad_variation = np.std(grad) / (np.mean(np.abs(grad)) + 1e-8)
                grad_variations.append(grad_variation)
            
            gradients.append(np.mean(grad_variations))
        
        return {
            'avg_gradient_variation': np.mean(gradients),
            'max_gradient_variation': np.max(gradients)
        }

class PCEOrderSelector:
    """PCE阶数选择器"""
    
    def __init__(self, X, Y, max_order=5):
        """
        初始化选择器
        
        Args:
            X: 输入数据
            Y: 输出数据
            max_order: 最大测试阶数
        """
        self.X = X
        self.Y = Y
        self.max_order = max_order
        self.analyzer = NonlinearityAnalyzer(X, Y)
        
    def select_optimal_order(self):
        """选择最优阶数"""
        print("🔍 分析函数非线性强度...")
        
        # 1. 非线性强度分析
        nonlinearity_metrics = self.analyzer.compute_nonlinearity_metrics()
        
        # 2. 基于非线性强度的初步建议
        theory_based_order = self._theory_based_order_selection(nonlinearity_metrics)
        
        # 3. 交叉验证选择
        cv_results = self._cross_validation_order_selection()
        
        # 4. 信息准则选择
        ic_results = self._information_criterion_selection()
        
        # 5. 综合决策
        optimal_order = self._make_final_decision(theory_based_order, cv_results, ic_results)
        
        return {
            'optimal_order': optimal_order,
            'nonlinearity_metrics': nonlinearity_metrics,
            'theory_based_order': theory_based_order,
            'cv_results': cv_results,
            'ic_results': ic_results
        }
    
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
        
        # 归一化指标
        nonlinearity_score = (
            (1 - linear_corr) * 0.2 +           # 线性相关性越低，非线性越强
            min(nonlin_ratio, 2.0) * 0.2 +      # 非线性比例
            min(moment_complexity, 5.0) * 0.15 + # 高阶矩复杂度
            min(high_freq_ratio, 1.0) * 0.2 +   # 高频成分
            (1 - local_linearity) * 0.15 +      # 局部非线性
            min(grad_variation, 3.0) * 0.1      # 梯度变化
        )
        
        print(f"   非线性强度分数: {nonlinearity_score:.3f}")
        
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
        
        print(f"   理论建议阶数: {suggested_order} ({reason})")
        
        return {
            'suggested_order': suggested_order,
            'nonlinearity_score': nonlinearity_score,
            'reason': reason,
            'detailed_scores': {
                'linear_correlation': linear_corr,
                'nonlinearity_ratio': nonlin_ratio,
                'moment_complexity': moment_complexity,
                'high_frequency_ratio': high_freq_ratio,
                'local_linearity': local_linearity,
                'gradient_variation': grad_variation
            }
        }
    
    def _cross_validation_order_selection(self):
        """交叉验证选择阶数"""
        print("🔄 交叉验证测试不同阶数...")
        
        orders = range(1, min(self.max_order + 1, 6))  # 限制最大阶数
        cv_scores = {}
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for order in orders:
            scores = []
            
            for train_idx, val_idx in kf.split(self.X):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                Y_train, Y_val = self.Y[train_idx], self.Y[val_idx]
                
                try:
                    # 创建简化的PCE模型
                    pce_score = self._fit_evaluate_pce(X_train, Y_train, X_val, Y_val, order)
                    scores.append(pce_score)
                except Exception as e:
                    print(f"   警告: 阶数{order}测试失败: {e}")
                    scores.append(0)
            
            cv_scores[order] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
            
            print(f"   阶数{order}: R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        # 选择最佳阶数
        best_order = max(cv_scores.keys(), key=lambda k: cv_scores[k]['mean_score'])
        
        return {
            'best_order': best_order,
            'cv_scores': cv_scores
        }
    
    def _fit_evaluate_pce(self, X_train, Y_train, X_val, Y_val, order):
        """拟合和评估PCE模型"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        # 创建多项式特征
        poly = PolynomialFeatures(degree=order, include_bias=True)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        # 训练模型
        model = Ridge(alpha=1e-6)
        
        if Y_train.ndim == 1:
            model.fit(X_train_poly, Y_train)
            Y_pred = model.predict(X_val_poly)
            return r2_score(Y_val, Y_pred)
        else:
            scores = []
            for i in range(Y_train.shape[1]):
                model.fit(X_train_poly, Y_train[:, i])
                Y_pred = model.predict(X_val_poly)
                score = r2_score(Y_val[:, i], Y_pred)
                scores.append(max(0, score))  # 避免负分数
            return np.mean(scores)
    
    def _information_criterion_selection(self):
        """信息准则选择"""
        print("📈 信息准则分析...")
        
        orders = range(1, min(self.max_order + 1, 6))
        ic_results = {}
        
        for order in orders:
            try:
                aic, bic = self._compute_information_criteria(order)
                ic_results[order] = {'AIC': aic, 'BIC': bic}
                print(f"   阶数{order}: AIC = {aic:.2f}, BIC = {bic:.2f}")
            except Exception as e:
                print(f"   警告: 阶数{order}信息准则计算失败: {e}")
                ic_results[order] = {'AIC': float('inf'), 'BIC': float('inf')}
        
        # 选择AIC和BIC最小的阶数
        best_aic_order = min(ic_results.keys(), key=lambda k: ic_results[k]['AIC'])
        best_bic_order = min(ic_results.keys(), key=lambda k: ic_results[k]['BIC'])
        
        return {
            'best_aic_order': best_aic_order,
            'best_bic_order': best_bic_order,
            'ic_results': ic_results
        }
    
    def _compute_information_criteria(self, order):
        """计算AIC和BIC"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        # 创建多项式特征
        poly = PolynomialFeatures(degree=order, include_bias=True)
        X_poly = poly.fit_transform(self.X)
        
        n_samples, n_features = X_poly.shape
        
        # 拟合模型并计算残差
        model = LinearRegression()
        
        if self.Y.ndim == 1:
            model.fit(X_poly, self.Y)
            Y_pred = model.predict(X_poly)
            mse = mean_squared_error(self.Y, Y_pred)
        else:
            mses = []
            for i in range(self.Y.shape[1]):
                model.fit(X_poly, self.Y[:, i])
                Y_pred = model.predict(X_poly)
                mse = mean_squared_error(self.Y[:, i], Y_pred)
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
        
        # 收集所有建议
        suggestions = [
            theory_result['suggested_order'],
            cv_result['best_order'],
            ic_result['best_aic_order'],
            ic_result['best_bic_order']
        ]
        
        # 计算加权平均（理论分析权重更高）
        weights = [0.4, 0.3, 0.15, 0.15]
        weighted_order = sum(s * w for s, w in zip(suggestions, weights))
        
        # 选择最接近的整数阶数
        final_order = int(round(weighted_order))
        
        # 确保在合理范围内
        final_order = max(1, min(final_order, self.max_order))
        
        print(f"   理论建议: {theory_result['suggested_order']}")
        print(f"   交叉验证: {cv_result['best_order']}")
        print(f"   AIC建议: {ic_result['best_aic_order']}")
        print(f"   BIC建议: {ic_result['best_bic_order']}")
        print(f"   加权平均: {weighted_order:.2f}")
        print(f"   最终决策: {final_order}")
        
        return final_order

def demonstrate_order_selection():
    """演示阶数选择过程"""
    print("🎯 PCE最优阶数选择演示")
    print("=" * 60)
    
    # 生成不同复杂度的测试函数
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (1000, 2))
    
    test_functions = {
        "线性函数": lambda x1, x2: 0.5 + 1.2*x1 + 0.8*x2,
        "二次函数": lambda x1, x2: 0.5 + 1.2*x1 + 0.8*x2 + 0.6*x1**2 + 0.4*x1*x2 + 0.3*x2**2,
        "三次函数": lambda x1, x2: 0.5 + x1 + x2 + x1**2 + x1*x2 + x2**2 + 0.3*x1**3 + 0.2*x1**2*x2,
        "复杂非线性": lambda x1, x2: np.sin(2*x1) * np.cos(2*x2) + 0.3*np.exp(-0.5*(x1**2 + x2**2))
    }
    
    results = {}
    
    for func_name, func in test_functions.items():
        print(f"\n{'='*20} {func_name} {'='*20}")
        
        # 生成数据
        Y = np.array([func(X[i, 0], X[i, 1]) for i in range(len(X))])
        Y = Y.reshape(-1, 1)
        
        # 选择最优阶数
        selector = PCEOrderSelector(X, Y, max_order=4)
        result = selector.select_optimal_order()
        
        results[func_name] = result
        
        print(f"✅ {func_name}最优阶数: {result['optimal_order']}")
    
    return results

if __name__ == "__main__":
    results = demonstrate_order_selection()
    
    print("\n" + "="*60)
    print("📊 总结报告")
    print("="*60)
    
    for func_name, result in results.items():
        print(f"\n{func_name}:")
        print(f"  最优阶数: {result['optimal_order']}")
        print(f"  非线性强度: {result['nonlinearity_metrics']['linear_correlation']['nonlinearity_ratio']:.3f}")
        print(f"  理论建议: {result['theory_based_order']['suggested_order']} ({result['theory_based_order']['reason']})")
