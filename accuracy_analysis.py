#!/usr/bin/env python3
"""
PCE精度分析 - 测试PCE在不同类型函数上的精度表现
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pce_trainer import PCETrainer
import warnings
warnings.filterwarnings('ignore')

def test_polynomial_function():
    """测试多项式函数 - PCE应该表现很好"""
    print("=" * 60)
    print("测试1: 多项式函数 (PCE优势场景)")
    print("=" * 60)
    
    def poly_func(x1, x2):
        """纯多项式函数"""
        results = []
        for i in range(78):
            # 2阶多项式组合
            y = (0.5 + i*0.01) + (0.3 + i*0.005)*x1 + (0.2 + i*0.003)*x2 + \
                (0.1 + i*0.002)*x1**2 + (0.15 + i*0.001)*x1*x2 + (0.08 + i*0.001)*x2**2
            results.append(y)
        return np.array(results)
    
    return test_function(poly_func, "多项式函数", noise_level=0.01)

def test_smooth_nonlinear():
    """测试平滑非线性函数 - PCE中等表现"""
    print("=" * 60)
    print("测试2: 平滑非线性函数 (PCE中等场景)")
    print("=" * 60)
    
    def smooth_func(x1, x2):
        """平滑的非线性函数"""
        results = []
        for i in range(78):
            # 包含三角函数和指数函数，但相对平滑
            y = 0.5 * np.sin(x1 + i*0.1) * np.cos(x2) + \
                0.3 * np.exp(-0.5*(x1**2 + x2**2)) + \
                0.2 * (x1**2 + x2**2) + i*0.01
            results.append(y)
        return np.array(results)
    
    return test_function(smooth_func, "平滑非线性函数", noise_level=0.01)

def test_complex_nonlinear():
    """测试复杂非线性函数 - PCE劣势场景"""
    print("=" * 60)
    print("测试3: 复杂非线性函数 (PCE劣势场景)")
    print("=" * 60)
    
    def complex_func(x1, x2):
        """复杂的非线性函数"""
        results = []
        for i in range(78):
            # 包含高频振荡、不连续性等
            y = np.sin(5*np.pi*x1) * np.cos(3*np.pi*x2) + \
                np.tanh(10*x1*x2) + \
                np.sign(x1 + x2) * np.sqrt(abs(x1*x2)) + \
                i*0.02
            results.append(y)
        return np.array(results)
    
    return test_function(complex_func, "复杂非线性函数", noise_level=0.01)

def test_function(func, func_name, noise_level=0.01, n_samples=1500):
    """测试指定函数的PCE vs NN性能"""
    
    # 生成数据
    X = np.random.uniform(-1, 1, (n_samples, 2))
    Y = np.array([func(x[0], x[1]) for x in X])
    Y += np.random.normal(0, noise_level, Y.shape)
    
    # 分割数据
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # 训练PCE
    print(f"训练PCE模型 ({func_name})...")
    pce_trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
    pce_trainer.train(X_train, Y_train, test_size=0.1)
    Y_pred_pce = pce_trainer.predict(X_test)
    
    # 训练神经网络
    print(f"训练神经网络模型 ({func_name})...")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    nn.fit(X_train_scaled, Y_train_scaled)
    Y_pred_nn_scaled = nn.predict(X_test_scaled)
    Y_pred_nn = scaler_Y.inverse_transform(Y_pred_nn_scaled)
    
    # 计算指标
    pce_mse = mean_squared_error(Y_test, Y_pred_pce)
    nn_mse = mean_squared_error(Y_test, Y_pred_nn)
    pce_r2 = r2_score(Y_test, Y_pred_pce)
    nn_r2 = r2_score(Y_test, Y_pred_nn)
    
    # 打印结果
    print(f"\n{func_name} 精度对比:")
    print(f"  PCE  - MSE: {pce_mse:.6f}, R²: {pce_r2:.6f}")
    print(f"  NN   - MSE: {nn_mse:.6f}, R²: {nn_r2:.6f}")
    print(f"  MSE比值 (NN/PCE): {nn_mse/pce_mse:.4f}")
    print(f"  R²差异 (NN-PCE): {nn_r2-pce_r2:.6f}")
    
    # 判断PCE表现
    if pce_r2 > 0.8:
        performance = "优秀"
    elif pce_r2 > 0.5:
        performance = "良好"
    elif pce_r2 > 0.2:
        performance = "中等"
    else:
        performance = "较差"
    
    print(f"  PCE表现评级: {performance}")
    
    return {
        'function_name': func_name,
        'pce_mse': pce_mse,
        'nn_mse': nn_mse,
        'pce_r2': pce_r2,
        'nn_r2': nn_r2,
        'performance': performance
    }

def visualize_accuracy_comparison(results):
    """可视化精度对比结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    functions = [r['function_name'] for r in results]
    pce_r2 = [r['pce_r2'] for r in results]
    nn_r2 = [r['nn_r2'] for r in results]
    
    x = np.arange(len(functions))
    width = 0.35
    
    # R²对比
    ax1.bar(x - width/2, pce_r2, width, label='PCE', alpha=0.8, color='blue')
    ax1.bar(x + width/2, nn_r2, width, label='Neural Network', alpha=0.8, color='red')
    ax1.set_xlabel('函数类型')
    ax1.set_ylabel('R² Score')
    ax1.set_title('PCE vs 神经网络 R²对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(functions, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE对比 (对数尺度)
    pce_mse = [r['pce_mse'] for r in results]
    nn_mse = [r['nn_mse'] for r in results]
    
    ax2.bar(x - width/2, pce_mse, width, label='PCE', alpha=0.8, color='blue')
    ax2.bar(x + width/2, nn_mse, width, label='Neural Network', alpha=0.8, color='red')
    ax2.set_xlabel('函数类型')
    ax2.set_ylabel('MSE (对数尺度)')
    ax2.set_title('PCE vs 神经网络 MSE对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(functions, rotation=45)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pce_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_accuracy_report(results):
    """生成精度分析报告"""
    report = f"""
{'='*80}
PCE精度分析报告
{'='*80}

测试概述:
本报告测试了PCE在不同类型函数上的精度表现，与神经网络进行对比。

测试结果:
"""
    
    for i, result in enumerate(results, 1):
        report += f"""
{i}. {result['function_name']}:
   PCE表现: {result['performance']}
   PCE  R²: {result['pce_r2']:.6f}  MSE: {result['pce_mse']:.6f}
   NN   R²: {result['nn_r2']:.6f}  MSE: {result['nn_mse']:.6f}
   精度差距: R²相差 {result['nn_r2']-result['pce_r2']:.3f}
"""
    
    report += f"""
{'='*80}
结论和建议:

1. PCE适用场景:
   ✅ 多项式函数或接近多项式的函数
   ✅ 平滑的低阶非线性函数
   ✅ 对精度要求不是极高，但对速度要求很高的场景

2. PCE不适用场景:
   ❌ 高频振荡函数
   ❌ 包含不连续性的函数
   ❌ 高阶复杂非线性函数
   ❌ 对精度要求极高的场景

3. 权衡建议:
   - 如果R² > 0.8: PCE是很好的选择，速度优势明显
   - 如果0.5 < R² < 0.8: 需要权衡精度vs速度
   - 如果R² < 0.5: 建议使用神经网络

4. 提高PCE精度的方法:
   - 增加训练数据量
   - 使用更高阶的多项式展开
   - 对输入进行预处理和特征工程
   - 使用分段PCE或多个PCE组合

{'='*80}
"""
    
    with open('pce_accuracy_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)

def main():
    """主函数"""
    print("PCE精度分析 - 测试不同类型函数")
    print("这将帮助您了解PCE在什么情况下精度可接受")
    
    results = []
    
    # 运行测试
    results.append(test_polynomial_function())
    results.append(test_smooth_nonlinear())
    results.append(test_complex_nonlinear())
    
    # 生成可视化和报告
    visualize_accuracy_comparison(results)
    generate_accuracy_report(results)
    
    print("\n" + "="*60)
    print("精度分析完成!")
    print("生成文件:")
    print("  - pce_accuracy_analysis.png: 精度对比图")
    print("  - pce_accuracy_report.txt: 详细分析报告")
    print("="*60)

if __name__ == "__main__":
    main()
