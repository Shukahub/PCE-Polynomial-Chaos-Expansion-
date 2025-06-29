#!/usr/bin/env python3
"""
PCE系统快速测试脚本
验证整个PCE系统是否正常工作
"""

import numpy as np
import os
import time
from pce_trainer import PCETrainer

def test_pce_system():
    """测试PCE系统的完整功能"""
    print("=" * 60)
    print("PCE神经网络替代系统 - 快速测试")
    print("=" * 60)
    
    # 测试1: 检查文件是否存在
    print("\n1. 检查关键文件...")
    required_files = [
        'pce_trainer.py',
        'PCE_fixed.f90', 
        'final_pce_model.pkl',
        'final_pce_coefficients.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} - 缺失!")
            return False
    
    # 测试2: 加载PCE模型
    print("\n2. 测试PCE模型加载...")
    try:
        trainer = PCETrainer()
        trainer.load_model('final_pce_model.pkl')
        print("   ✓ PCE模型加载成功")
    except Exception as e:
        print(f"   ✗ PCE模型加载失败: {e}")
        return False
    
    # 测试3: 测试推理功能
    print("\n3. 测试PCE推理功能...")
    try:
        test_inputs = np.array([
            [0.5, -0.3],
            [0.0, 0.0], 
            [-0.8, 0.9],
            [1.0, 1.0],
            [-1.0, -1.0]
        ])
        
        predictions = trainer.predict(test_inputs)
        
        if predictions.shape == (5, 78):
            print(f"   ✓ 推理成功，输出形状: {predictions.shape}")
            print(f"   ✓ 示例输出: {predictions[0, :3]}")
        else:
            print(f"   ✗ 输出形状错误: {predictions.shape}")
            return False
            
    except Exception as e:
        print(f"   ✗ 推理失败: {e}")
        return False
    
    # 测试4: 测试推理速度
    print("\n4. 测试推理速度...")
    try:
        test_data = np.random.uniform(-1, 1, (1000, 2))
        
        start_time = time.time()
        for _ in range(100):
            _ = trainer.predict(test_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = len(test_data) / avg_time
        
        print(f"   ✓ 平均推理时间: {avg_time*1000:.2f} ms (1000样本)")
        print(f"   ✓ 吞吐量: {throughput:.0f} 样本/秒")
        
    except Exception as e:
        print(f"   ✗ 速度测试失败: {e}")
        return False
    
    # 测试5: 检查Fortran可执行文件
    print("\n5. 检查Fortran程序...")
    fortran_executables = ['pce_demo.exe', 'pce_demo_final.exe']
    fortran_found = False
    
    for exe in fortran_executables:
        if os.path.exists(exe):
            print(f"   ✓ 找到Fortran可执行文件: {exe}")
            fortran_found = True
            break
    
    if not fortran_found:
        print("   ⚠ 未找到Fortran可执行文件，请运行: gfortran -O3 -o pce_demo PCE_fixed.f90")
    
    # 测试6: 验证系数文件格式
    print("\n6. 验证系数文件格式...")
    try:
        with open('final_pce_coefficients.txt', 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 80:  # 应该有注释行 + 78行系数
            print("   ✓ 系数文件格式正确")
        else:
            print(f"   ⚠ 系数文件行数可能不足: {len(lines)}")
            
    except Exception as e:
        print(f"   ✗ 系数文件验证失败: {e}")
        return False
    
    # 测试7: 内存使用测试
    print("\n7. 测试内存使用...")
    try:
        # 创建大批量数据测试内存使用
        large_batch = np.random.uniform(-1, 1, (10000, 2))
        predictions = trainer.predict(large_batch)
        
        if predictions.shape == (10000, 78):
            print("   ✓ 大批量处理成功")
        else:
            print(f"   ✗ 大批量处理输出形状错误: {predictions.shape}")
            return False
            
    except Exception as e:
        print(f"   ✗ 内存测试失败: {e}")
        return False
    
    return True

def print_system_info():
    """打印系统信息"""
    print("\n" + "=" * 60)
    print("系统信息总结")
    print("=" * 60)
    
    # 统计文件
    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    pce_files = [f for f in all_files if 'pce' in f.lower()]
    
    print(f"总文件数: {len(all_files)}")
    print(f"PCE相关文件: {len(pce_files)}")
    
    # 关键文件大小
    key_files = {
        'final_pce_model.pkl': 'Python PCE模型',
        'final_pce_coefficients.txt': 'Fortran系数文件',
        'performance_report.txt': '性能报告',
        'README.md': '使用说明'
    }
    
    print("\n关键文件:")
    for file, desc in key_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  {file:<30} ({desc}): {size:,} 字节")
    
    print(f"\n✅ PCE系统已完全部署并可用!")
    print("🚀 可以开始使用PCE替代神经网络进行快速推理!")

def main():
    """主函数"""
    success = test_pce_system()
    
    if success:
        print("\n" + "🎉" * 20)
        print("所有测试通过! PCE系统工作正常!")
        print("🎉" * 20)
        print_system_info()
    else:
        print("\n" + "❌" * 20)
        print("测试失败! 请检查系统配置!")
        print("❌" * 20)
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()
