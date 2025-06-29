#!/usr/bin/env python3
"""
PCE使用示例 - 展示如何在实际项目中使用PCE替代神经网络
"""

import numpy as np
import time
from pce_trainer import PCETrainer

def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("=" * 50)
    print("示例1: PCE基本使用方法")
    print("=" * 50)
    
    # 1. 创建PCE训练器
    trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
    
    # 2. 生成或加载训练数据
    print("生成训练数据...")
    X_train, Y_train = trainer.generate_training_data(n_samples=1000, noise_level=0.01)
    
    # 3. 训练PCE模型
    print("训练PCE模型...")
    results = trainer.train(X_train, Y_train)
    
    # 4. 进行预测
    print("进行预测...")
    test_input = np.array([[0.5, -0.3], [0.0, 0.0], [-0.8, 0.9]])
    predictions = trainer.predict(test_input)
    
    print(f"测试输入: {test_input}")
    print(f"预测输出形状: {predictions.shape}")
    print(f"前5个输出: {predictions[0, :5]}")
    
    # 5. 保存模型
    trainer.save_model('example_pce_model.pkl')
    trainer.export_fortran_coefficients('example_coefficients.txt')
    
    print("模型已保存!")

def example_2_load_and_use():
    """示例2: 加载已训练的模型"""
    print("\n" + "=" * 50)
    print("示例2: 加载并使用已训练的PCE模型")
    print("=" * 50)
    
    try:
        # 加载已训练的模型
        trainer = PCETrainer()
        trainer.load_model('final_pce_model.pkl')
        
        # 生成测试数据
        test_inputs = np.random.uniform(-1, 1, (100, 2))
        
        # 测试推理速度
        start_time = time.time()
        for _ in range(1000):
            predictions = trainer.predict(test_inputs)
        end_time = time.time()
        
        print(f"1000次推理用时: {end_time - start_time:.4f} 秒")
        print(f"平均每次推理: {(end_time - start_time) / 1000 * 1000:.4f} 毫秒")
        print(f"预测结果形状: {predictions.shape}")
        
    except FileNotFoundError:
        print("未找到已训练的模型文件，请先运行示例1或pce_demo.py")

def example_3_custom_function():
    """示例3: 训练自定义函数的PCE模型"""
    print("\n" + "=" * 50)
    print("示例3: 训练自定义函数的PCE模型")
    print("=" * 50)
    
    def custom_function(x1, x2):
        """自定义的复杂非线性函数"""
        results = []
        for i in range(78):
            # 为每个输出维度定义不同的函数
            if i < 26:
                # 三角函数组合
                y = np.sin(2 * np.pi * x1 + i * 0.1) * np.cos(np.pi * x2) + 0.5 * (x1**2 + x2**2)
            elif i < 52:
                # 指数函数组合
                y = np.exp(-0.5 * (x1**2 + x2**2)) * (x1 + x2) + 0.3 * x1 * x2
            else:
                # 多项式组合
                y = x1**3 - x2**3 + 0.5 * x1 * x2 + 0.1 * i
            
            results.append(y)
        
        return np.array(results)
    
    # 生成训练数据
    n_samples = 1500
    X = np.random.uniform(-1, 1, (n_samples, 2))
    Y = np.zeros((n_samples, 78))
    
    print("生成自定义函数的训练数据...")
    for i in range(n_samples):
        Y[i, :] = custom_function(X[i, 0], X[i, 1])
    
    # 添加噪声
    Y += np.random.normal(0, 0.01, Y.shape)
    
    # 训练PCE模型
    trainer = PCETrainer(input_dim=2, output_dim=78, polynomial_order=2)
    print("训练自定义函数的PCE模型...")
    results = trainer.train(X, Y)
    
    # 测试精度
    test_X = np.random.uniform(-1, 1, (100, 2))
    true_Y = np.array([custom_function(x[0], x[1]) for x in test_X])
    pred_Y = trainer.predict(test_X)
    
    mse = np.mean((true_Y - pred_Y)**2)
    print(f"测试MSE: {mse:.6f}")
    
    # 保存模型
    trainer.save_model('custom_pce_model.pkl')
    trainer.export_fortran_coefficients('custom_coefficients.txt')

def example_4_real_time_simulation():
    """示例4: 实时仿真应用"""
    print("\n" + "=" * 50)
    print("示例4: 实时仿真应用示例")
    print("=" * 50)
    
    try:
        # 加载模型
        trainer = PCETrainer()
        trainer.load_model('final_pce_model.pkl')
        
        print("模拟实时控制系统...")
        print("每个时间步需要快速计算78个输出...")
        
        # 模拟实时控制循环
        dt = 0.001  # 1ms时间步
        total_time = 1.0  # 总仿真时间1秒
        n_steps = int(total_time / dt)
        
        # 初始状态
        state = np.array([0.0, 0.0])
        
        start_time = time.time()
        
        for step in range(n_steps):
            # 模拟状态更新
            state[0] = 0.5 * np.sin(2 * np.pi * step * dt)
            state[1] = 0.3 * np.cos(np.pi * step * dt)
            
            # PCE快速推理
            outputs = trainer.predict(state.reshape(1, -1))
            
            # 这里可以使用outputs进行控制决策
            # control_signal = some_function(outputs)
            
            if step % 100 == 0:  # 每100步打印一次
                print(f"Step {step}: State={state}, First 3 outputs={outputs[0, :3]}")
        
        end_time = time.time()
        
        print(f"\n仿真完成!")
        print(f"总时间: {end_time - start_time:.4f} 秒")
        print(f"平均每步时间: {(end_time - start_time) / n_steps * 1000:.4f} 毫秒")
        print(f"实时性能: {'满足' if (end_time - start_time) / n_steps < dt else '不满足'} 1ms要求")
        
    except FileNotFoundError:
        print("未找到已训练的模型文件，请先运行pce_demo.py")

def example_5_batch_processing():
    """示例5: 批量处理应用"""
    print("\n" + "=" * 50)
    print("示例5: 批量处理应用示例")
    print("=" * 50)
    
    try:
        # 加载模型
        trainer = PCETrainer()
        trainer.load_model('final_pce_model.pkl')
        
        # 生成大批量数据
        batch_size = 10000
        print(f"处理 {batch_size} 个样本的批量数据...")
        
        batch_inputs = np.random.uniform(-1, 1, (batch_size, 2))
        
        # 批量推理
        start_time = time.time()
        batch_outputs = trainer.predict(batch_inputs)
        end_time = time.time()
        
        print(f"批量推理完成!")
        print(f"处理时间: {end_time - start_time:.4f} 秒")
        print(f"每个样本平均时间: {(end_time - start_time) / batch_size * 1000:.6f} 毫秒")
        print(f"吞吐量: {batch_size / (end_time - start_time):.0f} 样本/秒")
        
        # 保存结果
        np.savez('batch_results.npz', inputs=batch_inputs, outputs=batch_outputs)
        print("批量处理结果已保存到 batch_results.npz")
        
    except FileNotFoundError:
        print("未找到已训练的模型文件，请先运行pce_demo.py")

def main():
    """运行所有示例"""
    print("PCE使用示例集合")
    print("这些示例展示了如何在不同场景下使用PCE替代神经网络")
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_load_and_use()
    example_3_custom_function()
    example_4_real_time_simulation()
    example_5_batch_processing()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("生成的文件:")
    print("  - example_pce_model.pkl: 示例PCE模型")
    print("  - example_coefficients.txt: 示例Fortran系数")
    print("  - custom_pce_model.pkl: 自定义函数PCE模型")
    print("  - custom_coefficients.txt: 自定义函数Fortran系数")
    print("  - batch_results.npz: 批量处理结果")
    print("=" * 50)

if __name__ == "__main__":
    main()
