#!/usr/bin/env python3
"""
PCE训练时间和精度分析
测试不同配置下PCE的训练时间和精度表现
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pce_trainer import PCETrainer
import pandas as pd
import seaborn as sns

def test_training_time_vs_samples():
    """测试训练时间与样本数量的关系"""
    print("=" * 60)
    print("测试：训练时间 vs 样本数量")
    print("=" * 60)
    
    sample_sizes = [500, 1000, 2000, 5000, 10000]
    training_times = []
    test_accuracies = []
    
    for n_samples in sample_sizes:
        print(f"\n测试样本数量: {n_samples}")
        
        # 创建训练器
        trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
        
        # 生成数据
        X, Y = trainer.generate_training_data(n_samples=n_samples, noise_level=0.01)
        
        # 测量训练时间
        start_time = time.time()
        results = trainer.train(X, Y, test_size=0.2)
        training_time = time.time() - start_time
        
        training_times.append(training_time)
        test_accuracies.append(results['test_r2'])
        
        print(f"  训练时间: {training_time:.3f} 秒")
        print(f"  测试R²: {results['test_r2']:.6f}")
    
    return sample_sizes, training_times, test_accuracies

def test_polynomial_order_effect():
    """测试多项式阶数对精度和训练时间的影响"""
    print("=" * 60)
    print("测试：多项式阶数的影响")
    print("=" * 60)
    
    # 注意：高阶多项式需要修改PCETrainer代码支持
    # 这里只测试2阶，展示概念
    
    results = []
    
    # 固定样本数量
    n_samples = 2000
    
    # 测试不同的正则化参数（模拟不同复杂度）
    regularizations = [1e-8, 1e-6, 1e-4, 1e-2]
    
    for reg in regularizations:
        print(f"\n测试正则化参数: {reg}")
        
        trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
        X, Y = trainer.generate_training_data(n_samples=n_samples, noise_level=0.01)
        
        start_time = time.time()
        train_results = trainer.train(X, Y, regularization=reg)
        training_time = time.time() - start_time
        
        results.append({
            'regularization': reg,
            'training_time': training_time,
            'test_r2': train_results['test_r2'],
            'test_mse': train_results['test_mse']
        })
        
        print(f"  训练时间: {training_time:.3f} 秒")
        print(f"  测试R²: {train_results['test_r2']:.6f}")
        print(f"  测试MSE: {train_results['test_mse']:.6f}")
    
    return results

def compare_with_neural_network():
    """对比PCE和神经网络的训练时间"""
    print("=" * 60)
    print("对比：PCE vs 神经网络训练时间")
    print("=" * 60)
    
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    # 准备数据
    trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
    X, Y = trainer.generate_training_data(n_samples=2000, noise_level=0.01)
    
    # PCE训练
    print("\n训练PCE...")
    start_time = time.time()
    pce_results = trainer.train(X, Y)
    pce_time = time.time() - start_time
    
    # 神经网络训练
    print("\n训练神经网络...")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)
    
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    
    start_time = time.time()
    nn_model.fit(X_scaled, Y_scaled)
    nn_time = time.time() - start_time
    
    print(f"\nPCE训练时间: {pce_time:.3f} 秒")
    print(f"神经网络训练时间: {nn_time:.3f} 秒")
    print(f"速度提升: {nn_time/pce_time:.1f}x")
    
    print(f"\nPCE测试R²: {pce_results['test_r2']:.6f}")
    
    return pce_time, nn_time, pce_results['test_r2']

def create_training_analysis_plots():
    """创建训练分析图表"""
    
    # 测试1：样本数量 vs 训练时间
    sample_sizes, training_times, test_accuracies = test_training_time_vs_samples()
    
    # 测试2：正则化参数影响
    reg_results = test_polynomial_order_effect()
    
    # 测试3：与神经网络对比
    pce_time, nn_time, pce_r2 = compare_with_neural_network()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PCE Training Analysis', fontsize=16, fontweight='bold')
    
    # 图1：样本数量 vs 训练时间
    axes[0, 0].plot(sample_sizes, training_times, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Training Samples')
    axes[0, 0].set_ylabel('Training Time (seconds)')
    axes[0, 0].set_title('Training Time vs Sample Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加线性拟合
    z = np.polyfit(sample_sizes, training_times, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(sample_sizes, p(sample_sizes), "r--", alpha=0.8, 
                    label=f'Linear fit: {z[0]:.2e}x + {z[1]:.3f}')
    axes[0, 0].legend()
    
    # 图2：样本数量 vs 精度
    axes[0, 1].plot(sample_sizes, test_accuracies, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Training Samples')
    axes[0, 1].set_ylabel('Test R² Score')
    axes[0, 1].set_title('Accuracy vs Sample Size')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([min(test_accuracies) - 0.01, max(test_accuracies) + 0.01])
    
    # 图3：正则化参数影响
    reg_values = [r['regularization'] for r in reg_results]
    reg_r2_values = [r['test_r2'] for r in reg_results]
    reg_times = [r['training_time'] for r in reg_results]
    
    axes[1, 0].semilogx(reg_values, reg_r2_values, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Regularization Parameter')
    axes[1, 0].set_ylabel('Test R² Score')
    axes[1, 0].set_title('Regularization Effect on Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 图4：PCE vs NN训练时间对比
    methods = ['PCE', 'Neural Network']
    times = [pce_time, nn_time]
    colors = ['#2E86AB', '#A23B72']
    
    bars = axes[1, 1].bar(methods, times, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('PCE vs Neural Network Training Time')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # 添加速度提升标注
    axes[1, 1].text(0.5, max(times) * 0.8, f'PCE is {nn_time/pce_time:.1f}x faster', 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('pce_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_training_insights():
    """打印训练洞察"""
    print("\n" + "=" * 60)
    print("PCE训练特点总结")
    print("=" * 60)
    
    print("\n🚀 训练速度特点：")
    print("• PCE训练是一次性解析求解，不是迭代过程")
    print("• 训练时间与样本数量呈线性关系")
    print("• 典型训练时间：2000样本 < 1秒")
    print("• 比神经网络快10-100倍")
    
    print("\n🎯 精度提升方法：")
    print("• 增加训练样本数量（边际效应递减）")
    print("• 提高多项式阶数（需要修改代码）")
    print("• 调整正则化参数")
    print("• 改善数据质量（减少噪声）")
    
    print("\n❌ 无效的精度提升方法：")
    print("• 继续训练（PCE是解析解，一次确定）")
    print("• 调整学习率（PCE没有学习率概念）")
    print("• 早停策略（PCE不需要早停）")
    print("• 批次大小（PCE一次处理所有数据）")
    
    print("\n💡 关键洞察：")
    print("• PCE的精度上限由多项式阶数决定")
    print("• 对于复杂非线性函数，PCE精度有天然限制")
    print("• PCE的优势在于速度，不在于精度")
    print("• 选择PCE时要权衡精度与速度的需求")

def main():
    """主函数"""
    print("PCE Training Time and Accuracy Analysis")
    print("=" * 60)
    
    # 运行分析
    create_training_analysis_plots()
    
    # 打印洞察
    print_training_insights()
    
    print(f"\n分析完成！生成文件：pce_training_analysis.png")

if __name__ == "__main__":
    main()
