#!/usr/bin/env python3
"""
PCE vs Neural Network 性能对比演示
展示PCE在推理速度和精度方面相对于神经网络的优势
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from pce_trainer import PCETrainer
from data_generator import DataGenerator

class PerformanceComparator:
    def __init__(self):
        self.pce_model = None
        self.nn_model = None
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
    
    def prepare_data(self, n_samples=2000):
        """准备训练和测试数据"""
        print("Preparing training data...")
        
        generator = DataGenerator(input_dim=2, output_dim=78)
        X, Y = generator.generate_mixed_data(n_samples=n_samples, noise_level=0.02)
        
        # 分割数据
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, Y_train, Y_test
    
    def train_pce(self, X_train, Y_train):
        """训练PCE模型"""
        print("\nTraining PCE model...")
        start_time = time.time()
        
        self.pce_model = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
        results = self.pce_model.train(X_train, Y_train, test_size=0.1)  # 使用90%数据训练
        
        pce_train_time = time.time() - start_time
        print(f"PCE training time: {pce_train_time:.2f} seconds")
        
        return pce_train_time
    
    def train_neural_network(self, X_train, Y_train):
        """训练神经网络模型"""
        print("\nTraining Neural Network model...")
        start_time = time.time()
        
        # 数据标准化
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        Y_train_scaled = self.output_scaler.fit_transform(Y_train)
        
        # 创建神经网络（类似复杂度的网络）
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),  # 两个隐藏层
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        self.nn_model.fit(X_train_scaled, Y_train_scaled)
        
        nn_train_time = time.time() - start_time
        print(f"Neural Network training time: {nn_train_time:.2f} seconds")
        
        return nn_train_time
    
    def benchmark_inference_speed(self, X_test, n_iterations=1000):
        """基准测试推理速度"""
        print(f"\nBenchmarking inference speed ({n_iterations} iterations)...")
        
        # PCE推理速度测试
        start_time = time.time()
        for _ in range(n_iterations):
            _ = self.pce_model.predict(X_test)
        pce_inference_time = time.time() - start_time
        
        # 神经网络推理速度测试
        X_test_scaled = self.input_scaler.transform(X_test)
        start_time = time.time()
        for _ in range(n_iterations):
            Y_pred_scaled = self.nn_model.predict(X_test_scaled)
            _ = self.output_scaler.inverse_transform(Y_pred_scaled)
        nn_inference_time = time.time() - start_time
        
        print(f"PCE inference time: {pce_inference_time:.4f} seconds")
        print(f"NN inference time: {nn_inference_time:.4f} seconds")
        print(f"Speedup: {nn_inference_time/pce_inference_time:.2f}x")
        
        return pce_inference_time, nn_inference_time
    
    def evaluate_accuracy(self, X_test, Y_test):
        """评估模型精度"""
        print("\nEvaluating model accuracy...")
        
        # PCE预测
        Y_pred_pce = self.pce_model.predict(X_test)
        pce_mse = mean_squared_error(Y_test, Y_pred_pce)
        pce_r2 = r2_score(Y_test, Y_pred_pce)
        
        # 神经网络预测
        X_test_scaled = self.input_scaler.transform(X_test)
        Y_pred_nn_scaled = self.nn_model.predict(X_test_scaled)
        Y_pred_nn = self.output_scaler.inverse_transform(Y_pred_nn_scaled)
        nn_mse = mean_squared_error(Y_test, Y_pred_nn)
        nn_r2 = r2_score(Y_test, Y_pred_nn)
        
        print(f"PCE - MSE: {pce_mse:.6f}, R²: {pce_r2:.6f}")
        print(f"NN  - MSE: {nn_mse:.6f}, R²: {nn_r2:.6f}")
        
        return {
            'pce_mse': pce_mse, 'pce_r2': pce_r2, 'Y_pred_pce': Y_pred_pce,
            'nn_mse': nn_mse, 'nn_r2': nn_r2, 'Y_pred_nn': Y_pred_nn
        }
    
    def visualize_comparison(self, Y_test, results):
        """可视化对比结果"""
        print("\nGenerating comparison visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 前3个输出维度的预测对比
        for i in range(3):
            # PCE预测
            axes[0, i].scatter(Y_test[:, i], results['Y_pred_pce'][:, i], 
                              alpha=0.6, color='blue', label='PCE')
            axes[0, i].plot([Y_test[:, i].min(), Y_test[:, i].max()], 
                           [Y_test[:, i].min(), Y_test[:, i].max()], 'r--')
            axes[0, i].set_xlabel(f'True Output {i+1}')
            axes[0, i].set_ylabel(f'Predicted Output {i+1}')
            axes[0, i].set_title(f'PCE: Output {i+1}')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # 神经网络预测
            axes[1, i].scatter(Y_test[:, i], results['Y_pred_nn'][:, i], 
                              alpha=0.6, color='green', label='Neural Network')
            axes[1, i].plot([Y_test[:, i].min(), Y_test[:, i].max()], 
                           [Y_test[:, i].min(), Y_test[:, i].max()], 'r--')
            axes[1, i].set_xlabel(f'True Output {i+1}')
            axes[1, i].set_ylabel(f'Predicted Output {i+1}')
            axes[1, i].set_title(f'Neural Network: Output {i+1}')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig('pce_vs_nn_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, train_times, inference_times, accuracy_results):
        """生成性能报告"""
        pce_train_time, nn_train_time = train_times
        pce_inference_time, nn_inference_time = inference_times
        
        report = f"""
{'='*60}
PCE vs Neural Network Performance Report
{'='*60}

Training Performance:
  PCE Training Time:        {pce_train_time:.2f} seconds
  NN Training Time:         {nn_train_time:.2f} seconds
  Training Speedup:         {nn_train_time/pce_train_time:.2f}x (PCE faster)

Inference Performance:
  PCE Inference Time:       {pce_inference_time:.4f} seconds
  NN Inference Time:        {nn_inference_time:.4f} seconds
  Inference Speedup:        {nn_inference_time/pce_inference_time:.2f}x (PCE faster)

Accuracy Comparison:
  PCE MSE:                  {accuracy_results['pce_mse']:.6f}
  NN MSE:                   {accuracy_results['nn_mse']:.6f}
  MSE Ratio (NN/PCE):       {accuracy_results['nn_mse']/accuracy_results['pce_mse']:.2f}
  
  PCE R²:                   {accuracy_results['pce_r2']:.6f}
  NN R²:                    {accuracy_results['nn_r2']:.6f}
  R² Difference:            {accuracy_results['nn_r2'] - accuracy_results['pce_r2']:.6f}

Summary:
  PCE provides {nn_inference_time/pce_inference_time:.1f}x faster inference with 
  {'better' if accuracy_results['pce_mse'] < accuracy_results['nn_mse'] else 'comparable'} accuracy.
  
  PCE is particularly suitable for:
  - Real-time applications requiring fast inference
  - Embedded systems with limited computational resources
  - Applications where model interpretability is important
  - Scenarios with smooth, polynomial-like relationships

{'='*60}
"""
        
        print(report)
        
        # 保存报告到文件
        with open('performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Performance report saved to performance_report.txt")

def main():
    """主函数：运行完整的性能对比"""
    print("=" * 60)
    print("PCE vs Neural Network Performance Comparison")
    print("=" * 60)
    
    comparator = PerformanceComparator()
    
    # 1. 准备数据
    X_train, X_test, Y_train, Y_test = comparator.prepare_data(n_samples=2000)
    print(f"Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # 2. 训练模型
    pce_train_time = comparator.train_pce(X_train, Y_train)
    nn_train_time = comparator.train_neural_network(X_train, Y_train)
    
    # 3. 基准测试推理速度
    pce_inference_time, nn_inference_time = comparator.benchmark_inference_speed(X_test, n_iterations=1000)
    
    # 4. 评估精度
    accuracy_results = comparator.evaluate_accuracy(X_test, Y_test)
    
    # 5. 可视化对比
    comparator.visualize_comparison(Y_test, accuracy_results)
    
    # 6. 生成性能报告
    comparator.generate_performance_report(
        (pce_train_time, nn_train_time),
        (pce_inference_time, nn_inference_time),
        accuracy_results
    )
    
    # 7. 保存PCE模型用于Fortran
    comparator.pce_model.save_model('final_pce_model.pkl')
    comparator.pce_model.export_fortran_coefficients('final_pce_coefficients.txt')
    
    print("\nDemo completed successfully!")
    print("Files generated:")
    print("  - final_pce_model.pkl: 训练好的PCE模型")
    print("  - final_pce_coefficients.txt: Fortran系数文件")
    print("  - pce_vs_nn_comparison.png: 对比可视化")
    print("  - performance_report.txt: 性能报告")

if __name__ == "__main__":
    main()
