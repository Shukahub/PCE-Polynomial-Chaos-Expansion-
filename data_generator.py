#!/usr/bin/env python3
"""
数据生成器 - 为PCE训练生成各种类型的数据集
支持多种复杂的非线性函数来模拟真实的工程问题
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class DataGenerator:
    def __init__(self, input_dim=2, output_dim=78):
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def generate_engineering_data(self, n_samples=1000, noise_level=0.02):
        """
        生成模拟工程问题的数据集
        例如：结构响应、热传导、流体力学等
        """
        print(f"Generating engineering dataset with {n_samples} samples...")
        
        # 生成输入参数（例如：材料属性、几何参数等）
        X = np.random.uniform(-1, 1, (n_samples, self.input_dim))
        Y = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            x1, x2 = X[i, 0], X[i, 1]
            
            # 模拟结构响应：应力、应变、位移等
            for j in range(self.output_dim):
                # 基础响应函数
                if j < 26:  # 应力分量
                    response = (
                        0.8 * (x1**2 + x2**2) * np.exp(-0.3 * (x1**2 + x2**2)) +
                        0.4 * np.sin(3 * x1) * np.cos(2 * x2) +
                        0.2 * x1 * x2 * (1 + 0.1 * j)
                    )
                elif j < 52:  # 应变分量
                    response = (
                        0.6 * np.tanh(2 * x1) * np.sinh(x2) +
                        0.3 * (x1**3 - x2**3) * 0.1 +
                        0.1 * np.cos(np.pi * x1 * x2) * (1 + 0.05 * (j-26))
                    )
                else:  # 位移分量
                    response = (
                        0.5 * np.sin(np.pi * x1) * np.sin(np.pi * x2) +
                        0.3 * (x1 + x2) * np.exp(-0.5 * (x1**2 + x2**2)) +
                        0.2 * (x1**2 - x2**2) * (1 + 0.02 * (j-52))
                    )
                
                # 添加噪声
                Y[i, j] = response + np.random.normal(0, noise_level)
        
        return X, Y
    
    def generate_thermal_data(self, n_samples=1000, noise_level=0.01):
        """
        生成热传导问题的数据集
        """
        print(f"Generating thermal dataset with {n_samples} samples...")
        
        X = np.random.uniform(-1, 1, (n_samples, self.input_dim))
        Y = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            x1, x2 = X[i, 0], X[i, 1]  # 例如：热导率、边界条件
            
            for j in range(self.output_dim):
                # 模拟温度分布
                r = np.sqrt(x1**2 + x2**2)
                theta = np.arctan2(x2, x1)
                
                # 热传导解的近似
                temp_response = (
                    1.0 * np.exp(-r**2) * np.cos(theta + j * 0.08) +
                    0.5 * (1 - r**2) * np.sin(2 * theta) +
                    0.3 * x1 * x2 * np.exp(-0.5 * r) +
                    0.1 * j / self.output_dim  # 位置相关项
                )
                
                Y[i, j] = temp_response + np.random.normal(0, noise_level)
        
        return X, Y
    
    def generate_fluid_data(self, n_samples=1000, noise_level=0.015):
        """
        生成流体力学问题的数据集
        """
        print(f"Generating fluid dynamics dataset with {n_samples} samples...")
        
        X = np.random.uniform(-1, 1, (n_samples, self.input_dim))
        Y = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            x1, x2 = X[i, 0], X[i, 1]  # 例如：雷诺数、马赫数
            
            for j in range(self.output_dim):
                # 模拟流场变量：速度、压力、密度等
                if j < 26:  # 速度分量
                    response = (
                        0.7 * np.sin(2 * np.pi * x1) * (1 - x2**2) +
                        0.4 * x1 * (1 - x1**2) * np.cos(np.pi * x2) +
                        0.2 * np.tanh(x1 + x2) * (1 + 0.1 * j)
                    )
                elif j < 52:  # 压力分量
                    response = (
                        0.6 * (x1**2 + x2**2 - 1) * np.exp(-0.5 * (x1**2 + x2**2)) +
                        0.3 * np.sin(x1) * np.cos(x2) +
                        0.1 * (x1**3 + x2**3) * (1 + 0.05 * (j-26))
                    )
                else:  # 其他物理量
                    response = (
                        0.5 * np.cosh(x1) * np.sinh(x2) * 0.1 +
                        0.4 * (x1 - x2) * np.exp(-abs(x1 - x2)) +
                        0.2 * np.cos(3 * x1 + 2 * x2) * (1 + 0.02 * (j-52))
                    )
                
                Y[i, j] = response + np.random.normal(0, noise_level)
        
        return X, Y
    
    def generate_mixed_data(self, n_samples=1000, noise_level=0.02):
        """
        生成混合类型的复杂数据集
        """
        print(f"Generating mixed complex dataset with {n_samples} samples...")
        
        X = np.random.uniform(-1, 1, (n_samples, self.input_dim))
        Y = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            x1, x2 = X[i, 0], X[i, 1]
            
            for j in range(self.output_dim):
                # 复杂的非线性组合
                component1 = 0.4 * np.sin(2 * np.pi * x1 + j * 0.1) * np.cos(np.pi * x2)
                component2 = 0.3 * (x1**2 + x2**2) * np.exp(-0.5 * (x1**2 + x2**2))
                component3 = 0.2 * x1 * x2 * np.tanh(x1 + x2)
                component4 = 0.1 * (x1**3 - x2**3) * np.sin(j * 0.05)
                component5 = 0.05 * np.log(1 + abs(x1 + x2)) * np.cos(j * 0.03)
                
                # 添加输出维度相关的偏移和缩放
                scale_factor = 1 + 0.1 * np.sin(j * 0.02)
                offset = j * 0.01 * np.cos(x1 * x2)
                
                response = scale_factor * (component1 + component2 + component3 + component4 + component5) + offset
                
                Y[i, j] = response + np.random.normal(0, noise_level)
        
        return X, Y
    
    def visualize_data(self, X, Y, dataset_name="Dataset", save_plots=True):
        """
        可视化生成的数据集
        """
        print(f"Visualizing {dataset_name}...")
        
        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 输入数据分布
        plt.subplot(2, 4, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c='blue')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title('Input Data Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. 输出数据统计
        plt.subplot(2, 4, 2)
        output_means = np.mean(Y, axis=0)
        output_stds = np.std(Y, axis=0)
        plt.plot(output_means, 'b-', label='Mean')
        plt.fill_between(range(len(output_means)), 
                        output_means - output_stds, 
                        output_means + output_stds, 
                        alpha=0.3, label='±1 Std')
        plt.xlabel('Output Dimension')
        plt.ylabel('Value')
        plt.title('Output Statistics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3-6. 前4个输出维度的3D可视化
        for i in range(4):
            ax = fig.add_subplot(2, 4, i+3, projection='3d')
            scatter = ax.scatter(X[:, 0], X[:, 1], Y[:, i], 
                               c=Y[:, i], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Input 1')
            ax.set_ylabel('Input 2')
            ax.set_zlabel(f'Output {i+1}')
            ax.set_title(f'Output {i+1} Surface')
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        
        # 7. 输出相关性矩阵（前10个输出）
        plt.subplot(2, 4, 7)
        corr_matrix = np.corrcoef(Y[:, :10].T)
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.title('Output Correlation (First 10)')
        plt.xlabel('Output Index')
        plt.ylabel('Output Index')
        
        # 8. 输入-输出关系（第一个输出）
        plt.subplot(2, 4, 8)
        plt.scatter(X[:, 0], Y[:, 0], alpha=0.5, label='vs Input 1')
        plt.scatter(X[:, 1], Y[:, 0], alpha=0.5, label='vs Input 2')
        plt.xlabel('Input Value')
        plt.ylabel('Output 1')
        plt.title('Input-Output Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{dataset_name.lower().replace(' ', '_')}_visualization.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        
        plt.show()
    
    def save_data(self, X, Y, filename_prefix="dataset"):
        """
        保存数据集到文件
        """
        # 保存为numpy格式
        np.savez(f"{filename_prefix}.npz", X=X, Y=Y)
        
        # 保存为CSV格式
        data = np.column_stack([X, Y])
        columns = [f"Input_{i+1}" for i in range(self.input_dim)] + \
                 [f"Output_{i+1}" for i in range(self.output_dim)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f"{filename_prefix}.csv", index=False)
        
        print(f"Data saved to {filename_prefix}.npz and {filename_prefix}.csv")
        print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} inputs, {Y.shape[1]} outputs")

def main():
    """主函数：生成多种类型的数据集"""
    print("=" * 60)
    print("PCE Training Data Generator")
    print("=" * 60)
    
    generator = DataGenerator(input_dim=2, output_dim=78)
    
    # 生成不同类型的数据集
    datasets = [
        ("Engineering", generator.generate_engineering_data),
        ("Thermal", generator.generate_thermal_data),
        ("Fluid", generator.generate_fluid_data),
        ("Mixed Complex", generator.generate_mixed_data)
    ]
    
    for name, generate_func in datasets:
        print(f"\n{'-'*40}")
        print(f"Generating {name} Dataset")
        print(f"{'-'*40}")
        
        # 生成数据
        X, Y = generate_func(n_samples=1500, noise_level=0.02)
        
        # 可视化
        generator.visualize_data(X, Y, f"{name} Dataset", save_plots=True)
        
        # 保存数据
        generator.save_data(X, Y, f"{name.lower().replace(' ', '_')}_dataset")
        
        # 打印统计信息
        print(f"\nDataset Statistics:")
        print(f"  Input range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Output range: [{Y.min():.3f}, {Y.max():.3f}]")
        print(f"  Output mean: {Y.mean():.3f}")
        print(f"  Output std: {Y.std():.3f}")
    
    print(f"\n{'='*60}")
    print("All datasets generated successfully!")
    print("Files created:")
    print("  - *.npz: NumPy格式的数据文件")
    print("  - *.csv: CSV格式的数据文件")
    print("  - *_visualization.png: 数据可视化图片")

if __name__ == "__main__":
    main()
