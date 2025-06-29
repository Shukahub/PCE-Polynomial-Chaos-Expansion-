#!/usr/bin/env python3
"""
PCE (Polynomial Chaos Expansion) Trainer
用于训练PCE模型替代神经网络的Python程序

PCE使用多项式基函数来近似复杂的输入-输出关系
对于2输入78输出的问题，使用二阶多项式展开
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

class PCETrainer:
    def __init__(self, input_dim=2, output_dim=78, polynomial_order=2):
        """
        初始化PCE训练器
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度  
            polynomial_order: 多项式阶数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.polynomial_order = polynomial_order
        
        # 计算多项式基函数的数量
        # 对于2维输入，2阶多项式: 1, x1, x2, x1^2, x1*x2, x2^2 = 6个基函数
        self.n_basis = self._calculate_basis_count()
        
        # PCE系数矩阵 (output_dim x n_basis)
        self.coefficients = None
        
        # 数据标准化器
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        print(f"PCE Trainer initialized:")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Output dimension: {self.output_dim}")
        print(f"  Polynomial order: {self.polynomial_order}")
        print(f"  Number of basis functions: {self.n_basis}")
    
    def _calculate_basis_count(self):
        """计算多项式基函数数量"""
        if self.input_dim == 2 and self.polynomial_order == 2:
            return 6  # 1, x1, x2, x1^2, x1*x2, x2^2
        else:
            # 通用公式: C(n+d, d) where n=input_dim, d=polynomial_order
            from math import comb
            return comb(self.input_dim + self.polynomial_order, self.polynomial_order)
    
    def _compute_basis_functions(self, X):
        """
        计算多项式基函数
        
        Args:
            X: 输入数据 (n_samples, input_dim)
            
        Returns:
            basis_matrix: 基函数矩阵 (n_samples, n_basis)
        """
        n_samples = X.shape[0]
        basis_matrix = np.zeros((n_samples, self.n_basis))
        
        if self.input_dim == 2 and self.polynomial_order == 2:
            # 对于2D输入，2阶多项式
            x1, x2 = X[:, 0], X[:, 1]
            
            basis_matrix[:, 0] = 1.0        # 常数项
            basis_matrix[:, 1] = x1         # x1
            basis_matrix[:, 2] = x2         # x2
            basis_matrix[:, 3] = x1**2      # x1^2
            basis_matrix[:, 4] = x1 * x2    # x1*x2
            basis_matrix[:, 5] = x2**2      # x2^2
        else:
            raise NotImplementedError("目前只支持2D输入，2阶多项式")
        
        return basis_matrix
    
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
    
    def train(self, X, Y, test_size=0.2, regularization=1e-6):
        """
        训练PCE模型
        
        Args:
            X: 输入数据
            Y: 输出数据
            test_size: 测试集比例
            regularization: 正则化参数
        """
        print("Starting PCE training...")
        
        # 分割训练和测试数据
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )
        
        # 标准化数据
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_test_scaled = self.input_scaler.transform(X_test)
        Y_train_scaled = self.output_scaler.fit_transform(Y_train)
        Y_test_scaled = self.output_scaler.transform(Y_test)
        
        # 计算基函数矩阵
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
        print(f"  Training R²: {train_r2:.6f}")
        print(f"  Test R²: {test_r2:.6f}")
        
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
