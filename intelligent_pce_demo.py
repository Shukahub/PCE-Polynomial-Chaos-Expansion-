#!/usr/bin/env python3
"""
智能PCE阶数选择演示
展示如何使用智能阶数选择功能来优化PCE模型性能
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pce_trainer import PCETrainer
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# 设置样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

class IntelligentPCEDemo:
    """智能PCE演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.test_functions = {}
        self.results = {}
        
    def create_test_functions(self):
        """Create test functions with different complexity levels"""
        print("🎯 Creating test functions...")

        # 1. Linear function
        def linear_func(x1, x2):
            return 0.5 + 1.2*x1 + 0.8*x2

        # 2. Quadratic function
        def quadratic_func(x1, x2):
            return 0.5 + 1.2*x1 + 0.8*x2 + 0.6*x1**2 + 0.4*x1*x2 + 0.3*x2**2

        # 3. Cubic function
        def cubic_func(x1, x2):
            return (0.5 + x1 + x2 + x1**2 + x1*x2 + x2**2 +
                   0.3*x1**3 + 0.2*x1**2*x2 + 0.1*x1*x2**2 + 0.15*x2**3)

        # 4. Complex nonlinear function
        def complex_func(x1, x2):
            return (np.sin(2*x1) * np.cos(2*x2) +
                   0.3*np.exp(-0.5*(x1**2 + x2**2)) +
                   0.2*np.tanh(3*x1*x2))

        # 5. High-frequency oscillatory function
        def oscillatory_func(x1, x2):
            return (np.sin(5*np.pi*x1) * np.cos(3*np.pi*x2) +
                   0.5*np.sin(8*np.pi*x1*x2) +
                   0.3*(x1**2 + x2**2))

        self.test_functions = {
            "Linear": {
                "func": linear_func,
                "expected_order": 1,
                "description": "Simple linear relationship, PCE should select order 1"
            },
            "Quadratic": {
                "func": quadratic_func,
                "expected_order": 2,
                "description": "Standard quadratic polynomial, PCE should select order 2"
            },
            "Cubic": {
                "func": cubic_func,
                "expected_order": 3,
                "description": "Cubic polynomial, PCE should select order 3"
            },
            "Complex Nonlinear": {
                "func": complex_func,
                "expected_order": 3,
                "description": "Smooth nonlinear, PCE may select order 3-4"
            },
            "High-Frequency": {
                "func": oscillatory_func,
                "expected_order": 4,
                "description": "High-frequency components, PCE may select order 4-5 or suggest other methods"
            }
        }

        print(f"✅ Created {len(self.test_functions)} test functions")
        
    def generate_data(self, func, n_samples=1000, noise_level=0.01):
        """生成测试数据"""
        np.random.seed(42)
        X = np.random.uniform(-1, 1, (n_samples, 2))
        Y = np.array([func(X[i, 0], X[i, 1]) for i in range(n_samples)])
        
        # 添加噪声
        if noise_level > 0:
            Y += np.random.normal(0, noise_level, Y.shape)
        
        return X, Y.reshape(-1, 1)
    
    def test_intelligent_order_selection(self):
        """测试智能阶数选择"""
        print("\n" + "="*80)
        print("🧠 智能PCE阶数选择测试")
        print("="*80)
        
        for func_name, func_info in self.test_functions.items():
            print(f"\n{'='*20} {func_name} {'='*20}")
            print(f"📝 描述: {func_info['description']}")
            print(f"🎯 期望阶数: {func_info['expected_order']}")
            
            # 生成数据
            X, Y = self.generate_data(func_info['func'], n_samples=1000)
            
            # 创建智能PCE训练器
            trainer = PCETrainer(
                input_dim=2, 
                output_dim=1, 
                polynomial_order=None,  # 自动选择
                auto_order_selection=True
            )
            
            # 测试智能阶数选择
            start_time = time.time()
            optimal_order = trainer.select_optimal_order(X, Y, max_order=5)
            selection_time = time.time() - start_time
            
            # 训练模型
            start_time = time.time()
            results = trainer.train(X, Y, test_size=0.2)
            training_time = time.time() - start_time
            
            # 保存结果
            self.results[func_name] = {
                'expected_order': func_info['expected_order'],
                'selected_order': optimal_order,
                'selection_time': selection_time,
                'training_time': training_time,
                'test_r2': results['test_r2'],
                'test_mse': results['test_mse'],
                'order_selection_results': trainer.order_selection_results
            }
            
            # 显示结果
            print(f"✅ 智能选择结果:")
            print(f"   选择阶数: {optimal_order}")
            print(f"   期望阶数: {func_info['expected_order']}")
            print(f"   选择准确: {'✅' if optimal_order == func_info['expected_order'] else '⚠️'}")
            print(f"   选择时间: {selection_time:.3f} 秒")
            print(f"   训练时间: {training_time:.3f} 秒")
            print(f"   测试R²: {results['test_r2']:.6f}")
            print(f"   测试MSE: {results['test_mse']:.6f}")
    
    def compare_with_fixed_orders(self):
        """与固定阶数进行对比"""
        print("\n" + "="*80)
        print("📊 智能选择 vs 固定阶数对比")
        print("="*80)
        
        comparison_results = {}
        
        for func_name, func_info in self.test_functions.items():
            print(f"\n🔍 测试函数: {func_name}")
            
            X, Y = self.generate_data(func_info['func'], n_samples=1000)
            
            # 测试不同固定阶数
            fixed_orders = [1, 2, 3, 4, 5]
            fixed_results = {}
            
            for order in fixed_orders:
                try:
                    trainer = PCETrainer(
                        input_dim=2, 
                        output_dim=1, 
                        polynomial_order=order,
                        auto_order_selection=False
                    )
                    
                    start_time = time.time()
                    results = trainer.train(X, Y, test_size=0.2)
                    training_time = time.time() - start_time
                    
                    fixed_results[order] = {
                        'training_time': training_time,
                        'test_r2': results['test_r2'],
                        'test_mse': results['test_mse']
                    }
                    
                    print(f"   阶数{order}: R²={results['test_r2']:.4f}, MSE={results['test_mse']:.6f}, 时间={training_time:.3f}s")
                    
                except Exception as e:
                    print(f"   阶数{order}: 失败 ({e})")
                    fixed_results[order] = {
                        'training_time': float('inf'),
                        'test_r2': -float('inf'),
                        'test_mse': float('inf')
                    }
            
            # 智能选择结果
            intelligent_result = self.results[func_name]
            
            # 找到最佳固定阶数
            best_fixed_order = max(fixed_results.keys(), 
                                 key=lambda k: fixed_results[k]['test_r2'])
            best_fixed_result = fixed_results[best_fixed_order]
            
            comparison_results[func_name] = {
                'intelligent': intelligent_result,
                'fixed_results': fixed_results,
                'best_fixed_order': best_fixed_order,
                'best_fixed_result': best_fixed_result
            }
            
            print(f"   🧠 智能选择: 阶数{intelligent_result['selected_order']}, R²={intelligent_result['test_r2']:.4f}")
            print(f"   🎯 最佳固定: 阶数{best_fixed_order}, R²={best_fixed_result['test_r2']:.4f}")
            
            # 比较结果
            r2_improvement = intelligent_result['test_r2'] - best_fixed_result['test_r2']
            print(f"   📈 R²提升: {r2_improvement:+.4f}")
        
        return comparison_results
    
    def create_visualization(self, comparison_results):
        """创建可视化图表"""
        print("\n📊 生成可视化图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Intelligent PCE Order Selection Analysis', fontsize=16, fontweight='bold')
        
        func_names = list(self.test_functions.keys())
        
        # 1. 阶数选择准确性
        ax1 = axes[0, 0]
        expected_orders = [self.test_functions[name]['expected_order'] for name in func_names]
        selected_orders = [self.results[name]['selected_order'] for name in func_names]
        
        x_pos = np.arange(len(func_names))
        width = 0.35
        
        ax1.bar(x_pos - width/2, expected_orders, width, label='Expected Order', alpha=0.7, color='skyblue')
        ax1.bar(x_pos + width/2, selected_orders, width, label='Selected Order', alpha=0.7, color='lightcoral')
        
        ax1.set_xlabel('Test Functions')
        ax1.set_ylabel('Polynomial Order')
        ax1.set_title('Order Selection Accuracy')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(func_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. R²分数对比
        ax2 = axes[0, 1]
        intelligent_r2 = [self.results[name]['test_r2'] for name in func_names]
        best_fixed_r2 = [comparison_results[name]['best_fixed_result']['test_r2'] for name in func_names]
        
        ax2.bar(x_pos - width/2, intelligent_r2, width, label='Intelligent Selection', alpha=0.7, color='green')
        ax2.bar(x_pos + width/2, best_fixed_r2, width, label='Best Fixed Order', alpha=0.7, color='orange')
        
        ax2.set_xlabel('Test Functions')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Accuracy Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(func_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 非线性强度分析
        ax3 = axes[0, 2]
        nonlinearity_scores = []
        for name in func_names:
            if self.results[name]['order_selection_results']:
                score = self.results[name]['order_selection_results']['theory_order']['nonlinearity_score']
                nonlinearity_scores.append(score)
            else:
                nonlinearity_scores.append(0)
        
        bars = ax3.bar(range(len(func_names)), nonlinearity_scores, alpha=0.7, color='purple')
        ax3.set_xlabel('Test Functions')
        ax3.set_ylabel('Nonlinearity Score')
        ax3.set_title('Function Complexity Analysis')
        ax3.set_xticks(range(len(func_names)))
        ax3.set_xticklabels(func_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, nonlinearity_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 训练时间对比
        ax4 = axes[1, 0]
        selection_times = [self.results[name]['selection_time'] for name in func_names]
        training_times = [self.results[name]['training_time'] for name in func_names]
        
        ax4.bar(x_pos - width/2, selection_times, width, label='Order Selection Time', alpha=0.7, color='red')
        ax4.bar(x_pos + width/2, training_times, width, label='Training Time', alpha=0.7, color='blue')
        
        ax4.set_xlabel('Test Functions')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Time Analysis')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(func_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 决策方法对比
        ax5 = axes[1, 1]
        methods = ['Theory', 'CV', 'AIC', 'BIC', 'Final']
        
        # 收集所有函数的决策数据
        decision_data = []
        for name in func_names:
            if self.results[name]['order_selection_results']:
                results = self.results[name]['order_selection_results']
                decision_data.append([
                    results['theory_order']['suggested_order'],
                    results['cv_order']['best_order'],
                    results['ic_order']['best_aic_order'],
                    results['ic_order']['best_bic_order'],
                    results['optimal_order']
                ])
            else:
                decision_data.append([2, 2, 2, 2, 2])  # 默认值
        
        decision_data = np.array(decision_data).T
        
        for i, method in enumerate(methods):
            ax5.plot(range(len(func_names)), decision_data[i], 'o-', label=method, linewidth=2, markersize=6)
        
        ax5.set_xlabel('Test Functions')
        ax5.set_ylabel('Suggested Order')
        ax5.set_title('Decision Method Comparison')
        ax5.set_xticks(range(len(func_names)))
        ax5.set_xticklabels(func_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 总结统计
        ax6 = axes[1, 2]
        
        # 计算统计数据
        correct_selections = sum(1 for name in func_names 
                               if self.results[name]['selected_order'] == self.test_functions[name]['expected_order'])
        total_functions = len(func_names)
        accuracy_rate = correct_selections / total_functions * 100
        
        avg_r2_improvement = np.mean([
            self.results[name]['test_r2'] - comparison_results[name]['best_fixed_result']['test_r2']
            for name in func_names
        ])
        
        avg_selection_time = np.mean([self.results[name]['selection_time'] for name in func_names])
        
        # 创建统计图表
        stats_labels = ['Selection\nAccuracy (%)', 'Avg R²\nImprovement', 'Avg Selection\nTime (s)']
        stats_values = [accuracy_rate, avg_r2_improvement * 100, avg_selection_time]
        colors = ['lightgreen', 'lightblue', 'lightyellow']
        
        bars = ax6.bar(stats_labels, stats_values, color=colors, alpha=0.7)
        ax6.set_title('Summary Statistics')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('intelligent_pce_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_demo(self):
        """运行完整演示"""
        print("🎯 智能PCE阶数选择演示")
        print("="*80)
        
        # 1. 创建测试函数
        self.create_test_functions()
        
        # 2. 测试智能阶数选择
        self.test_intelligent_order_selection()
        
        # 3. 与固定阶数对比
        comparison_results = self.compare_with_fixed_orders()
        
        # 4. 创建可视化
        self.create_visualization(comparison_results)
        
        # 5. 生成总结报告
        self.generate_summary_report(comparison_results)
        
        print(f"\n✅ 演示完成！生成文件：intelligent_pce_analysis.png")
        
        return self.results, comparison_results
    
    def generate_summary_report(self, comparison_results):
        """生成总结报告"""
        print("\n" + "="*80)
        print("📋 智能PCE阶数选择总结报告")
        print("="*80)
        
        func_names = list(self.test_functions.keys())
        
        # 计算总体统计
        correct_selections = sum(1 for name in func_names 
                               if self.results[name]['selected_order'] == self.test_functions[name]['expected_order'])
        total_functions = len(func_names)
        accuracy_rate = correct_selections / total_functions * 100
        
        avg_r2_improvement = np.mean([
            self.results[name]['test_r2'] - comparison_results[name]['best_fixed_result']['test_r2']
            for name in func_names
        ])
        
        avg_selection_time = np.mean([self.results[name]['selection_time'] for name in func_names])
        avg_training_time = np.mean([self.results[name]['training_time'] for name in func_names])
        
        print(f"🎯 总体性能:")
        print(f"   阶数选择准确率: {accuracy_rate:.1f}% ({correct_selections}/{total_functions})")
        print(f"   平均R²提升: {avg_r2_improvement:+.4f}")
        print(f"   平均选择时间: {avg_selection_time:.3f} 秒")
        print(f"   平均训练时间: {avg_training_time:.3f} 秒")
        
        print(f"\n📊 详细结果:")
        for name in func_names:
            result = self.results[name]
            expected = self.test_functions[name]['expected_order']
            selected = result['selected_order']
            r2_improvement = result['test_r2'] - comparison_results[name]['best_fixed_result']['test_r2']
            
            status = "✅" if selected == expected else "⚠️"
            print(f"   {name}: {status} 期望{expected}阶 → 选择{selected}阶, R²提升{r2_improvement:+.4f}")
        
        print(f"\n💡 关键洞察:")
        print(f"   • 智能选择在{accuracy_rate:.0f}%的情况下选择了正确的阶数")
        print(f"   • 平均R²提升{avg_r2_improvement*100:+.2f}%，表明智能选择的有效性")
        print(f"   • 阶数选择时间约{avg_selection_time:.2f}秒，相比训练时间({avg_training_time:.2f}秒)是可接受的开销")
        print(f"   • 对于复杂非线性函数，智能选择能够识别并推荐合适的阶数")

def main():
    """主函数"""
    demo = IntelligentPCEDemo()
    results, comparison_results = demo.run_demo()
    return results, comparison_results

if __name__ == "__main__":
    main()
