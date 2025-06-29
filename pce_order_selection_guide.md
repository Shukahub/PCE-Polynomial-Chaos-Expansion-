# PCE最优阶数选择指南

## 📚 理论基础

### 1. 非线性强度分析方法

#### 🔍 **多维度非线性指标**

| 指标类型 | 计算方法 | 物理意义 | 权重 |
|---------|---------|---------|------|
| **线性相关性** | `corr(X, Y)` | 输入与输出的线性关系强度 | 20% |
| **非线性比例** | `corr(X²,Y) / corr(X,Y)` | 二次项相对于线性项的重要性 | 20% |
| **高阶矩复杂度** | `|skewness| + |kurtosis|` | 输出分布的非正态性 | 15% |
| **高频成分** | `FFT高频功率比` | 函数的振荡特性 | 20% |
| **局部线性度** | `局部R²平均值` | 函数在局部区域的线性程度 | 15% |
| **梯度变化** | `∇f变化的标准差` | 函数斜率的变化剧烈程度 | 10% |

#### 📊 **综合非线性强度分数**

```python
nonlinearity_score = (
    (1 - linear_corr) * 0.2 +           # 线性相关性越低，非线性越强
    min(nonlin_ratio, 2.0) * 0.2 +      # 非线性比例（截断避免异常值）
    min(moment_complexity, 5.0) * 0.15 + # 高阶矩复杂度
    min(high_freq_ratio, 1.0) * 0.2 +   # 高频成分
    (1 - local_linearity) * 0.15 +      # 局部非线性
    min(grad_variation, 3.0) * 0.1      # 梯度变化
)
```

### 2. 阶数选择决策树

```
非线性强度分数 < 0.3  →  阶数 1 (线性)
0.3 ≤ 分数 < 0.6     →  阶数 2 (二次)
0.6 ≤ 分数 < 1.0     →  阶数 3 (三次)
1.0 ≤ 分数 < 1.5     →  阶数 4 (四次)
分数 ≥ 1.5           →  阶数 5 (五次或考虑其他方法)
```

## 🎯 实用选择公式

### 快速估算公式

对于2维输入问题，可以使用以下简化公式：

```python
def quick_order_estimate(X, Y):
    """快速估算PCE阶数"""
    
    # 1. 计算线性相关性
    linear_r2 = max([r2_score(Y, X[:, i]) for i in range(X.shape[1])])
    
    # 2. 计算二次项相关性
    quad_features = [X[:, i]**2 for i in range(X.shape[1])]
    quad_features.extend([X[:, i]*X[:, j] for i in range(X.shape[1]) 
                         for j in range(i+1, X.shape[1])])
    
    quad_r2 = max([r2_score(Y, feat) for feat in quad_features])
    
    # 3. 决策逻辑
    if linear_r2 > 0.9:
        return 1  # 线性足够
    elif quad_r2 > 0.85:
        return 2  # 二次足够
    elif quad_r2 > 0.7:
        return 3  # 需要三次
    else:
        return 4  # 高阶或考虑其他方法
```

### 样本数量要求

PCE阶数选择还需要考虑样本数量限制：

```python
def check_sample_requirement(n_samples, n_features, order):
    """检查样本数量是否足够"""
    from math import comb
    
    # 计算基函数数量
    n_basis = comb(n_features + order, order)
    
    # 经验法则：样本数量应该是基函数数量的5-10倍
    min_samples = n_basis * 5
    recommended_samples = n_basis * 10
    
    if n_samples < min_samples:
        return "样本不足", min_samples
    elif n_samples < recommended_samples:
        return "样本勉强", recommended_samples
    else:
        return "样本充足", recommended_samples
```

## 📈 信息准则方法

### AIC (Akaike Information Criterion)

```python
AIC = 2k - 2ln(L)
```

其中：
- `k` = 参数数量（基函数数量）
- `L` = 似然函数值

### BIC (Bayesian Information Criterion)

```python
BIC = k*ln(n) - 2ln(L)
```

其中：
- `n` = 样本数量
- BIC对模型复杂度的惩罚更严格

### 使用建议

1. **AIC**: 更注重预测精度，适合预测导向的应用
2. **BIC**: 更注重模型简洁性，适合解释导向的应用
3. **实践中**: 通常选择AIC和BIC都较小的阶数

## 🔬 交叉验证方法

### K折交叉验证

```python
def cv_order_selection(X, Y, max_order=5, cv_folds=5):
    """交叉验证选择阶数"""
    
    best_order = 1
    best_score = -np.inf
    
    for order in range(1, max_order + 1):
        scores = []
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            # 训练和验证
            score = fit_and_evaluate_pce(X[train_idx], Y[train_idx], 
                                       X[val_idx], Y[val_idx], order)
            scores.append(score)
        
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_order = order
    
    return best_order, best_score
```

## 🎨 可视化诊断

### 残差分析

```python
def plot_residual_analysis(Y_true, Y_pred, order):
    """绘制残差分析图"""
    
    residuals = Y_true - Y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 残差 vs 预测值
    axes[0,0].scatter(Y_pred, residuals, alpha=0.6)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_title(f'Residuals vs Predicted (Order {order})')
    
    # 2. 残差直方图
    axes[0,1].hist(residuals, bins=30, alpha=0.7)
    axes[0,1].set_title('Residual Distribution')
    
    # 3. Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot')
    
    # 4. 残差自相关
    from statsmodels.tsa.stattools import acf
    lags = range(1, min(20, len(residuals)//4))
    autocorr = [acf(residuals, nlags=lag)[-1] for lag in lags]
    axes[1,1].plot(lags, autocorr, 'o-')
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_title('Residual Autocorrelation')
    
    plt.tight_layout()
    return fig
```

## 🚀 实际应用指南

### 1. 工程仿真场景

```python
# 结构力学、流体力学等
if problem_type == "engineering_simulation":
    if physics_based_smooth:
        suggested_order = 2  # 物理规律通常是平滑的
    elif has_discontinuities:
        suggested_order = 4  # 或考虑分段PCE
    else:
        suggested_order = 3
```

### 2. 控制系统场景

```python
# 实时控制系统
if problem_type == "control_system":
    if real_time_requirement:
        max_order = 3  # 平衡精度和速度
    if system_linear_range:
        suggested_order = 2
    else:
        suggested_order = 3
```

### 3. 金融建模场景

```python
# 金融风险建模
if problem_type == "financial_modeling":
    if market_volatility == "high":
        suggested_order = 4  # 捕捉复杂非线性
    elif interpretability_required:
        suggested_order = 2  # 保持可解释性
    else:
        suggested_order = 3
```

## ⚠️ 常见陷阱和注意事项

### 1. 过拟合风险

- **症状**: 训练精度很高，验证精度很低
- **原因**: 阶数过高，样本不足
- **解决**: 降低阶数或增加样本

### 2. 数值不稳定

- **症状**: 系数矩阵条件数过大
- **原因**: 高阶多项式基函数线性相关
- **解决**: 增加正则化或使用正交多项式

### 3. 外推能力差

- **症状**: 训练域内精度高，域外精度低
- **原因**: 高阶多项式外推不稳定
- **解决**: 限制阶数，扩大训练域

## 📋 决策检查清单

在选择PCE阶数时，请检查以下项目：

- [ ] 计算了非线性强度指标
- [ ] 进行了交叉验证测试
- [ ] 检查了样本数量充足性
- [ ] 考虑了计算资源限制
- [ ] 评估了外推需求
- [ ] 分析了残差分布
- [ ] 比较了AIC/BIC准则
- [ ] 考虑了应用场景特点

## 🎯 总结

选择PCE最优阶数是一个多因素决策过程：

1. **理论分析** - 基于非线性强度指标
2. **实证验证** - 交叉验证和信息准则
3. **实际约束** - 样本数量和计算资源
4. **应用需求** - 精度、速度、可解释性平衡

**推荐流程**：
1. 使用非线性强度分析获得初步建议
2. 交叉验证确认最佳性能阶数
3. 信息准则平衡复杂度和拟合度
4. 综合考虑实际应用约束
5. 残差分析验证模型适用性
