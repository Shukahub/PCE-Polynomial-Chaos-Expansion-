#!/usr/bin/env python3
"""
PCE项目演示脚本
展示如何快速使用PCE对比图表生成工具
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """打印步骤"""
    print(f"\n📋 步骤 {step}: {description}")
    print("-" * 40)

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"💻 执行: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"💥 {description} 异常: {e}")
        return False

def check_files():
    """检查关键文件"""
    print_step(1, "检查项目文件")
    
    required_files = [
        'pce_trainer.py',
        'generate_comparison_charts.py', 
        'generate_deployment_charts.py',
        'generate_all_charts.py',
        'view_charts.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 缺少关键文件: {', '.join(missing_files)}")
        return False
    
    print("\n✅ 所有关键文件都存在")
    return True

def check_dependencies():
    """检查Python依赖"""
    print_step(2, "检查Python依赖")
    
    packages = ['numpy', 'matplotlib', 'sklearn', 'pandas']
    missing = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ 缺少依赖包: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ 所有依赖都已安装")
    return True

def generate_charts():
    """生成对比图表"""
    print_step(3, "生成PCE对比图表")
    
    print("🎨 开始生成图表...")
    success = run_command("python generate_all_charts.py", "图表生成")
    
    if success:
        # 检查生成的图表文件
        chart_files = [
            'comprehensive_accuracy_comparison.png',
            'comprehensive_speed_comparison.png',
            'model_size_comparison.png',
            'memory_usage_comparison.png',
            'deployment_complexity_comparison.png',
            'platform_compatibility_comparison.png'
        ]
        
        print("\n📊 检查生成的图表:")
        found = 0
        for file in chart_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} bytes)")
                found += 1
            else:
                print(f"   ❌ {file}")
        
        print(f"\n📈 成功生成 {found}/{len(chart_files)} 个图表")
        return found > 0
    
    return False

def view_charts():
    """查看生成的图表"""
    print_step(4, "查看生成的图表")
    
    print("📋 列出所有图表文件:")
    success = run_command("python view_charts.py list", "图表列表")
    
    if success:
        print("\n💡 您可以使用以下命令查看图表:")
        print("   python view_charts.py           # 查看所有图表")
        print("   python view_charts.py accuracy  # 查看精度对比")
        print("   python view_charts.py speed     # 查看速度对比")
        print("   python view_charts.py deployment # 查看部署对比")
    
    return success

def demo_pce_usage():
    """演示PCE基本使用"""
    print_step(5, "PCE基本使用演示")
    
    print("🧮 演示PCE训练和推理...")
    
    demo_code = '''
import numpy as np
from pce_trainer import PCETrainer

# 创建PCE训练器
trainer = PCETrainer(input_dim=2, output_dim=3, polynomial_order=2)

# 生成示例数据
X = np.random.uniform(-1, 1, (100, 2))
Y_base = X[:, 0]**2 + X[:, 1]**2
Y = np.column_stack([Y_base + np.random.normal(0, 0.1, 100) for _ in range(3)])

# 训练模型
trainer.train(X, Y)

# 进行推理
test_input = np.array([[0.5, -0.3]])
prediction = trainer.predict(test_input)

print(f"输入: {test_input[0]}")
print(f"预测: {prediction[0]}")
print("✅ PCE演示完成!")
'''
    
    try:
        exec(demo_code)
        return True
    except Exception as e:
        print(f"❌ PCE演示失败: {e}")
        return False

def main():
    """主演示函数"""
    print_header("PCE神经网络替代方案 - 快速演示")
    
    print("🎯 本演示将展示:")
    print("   1. 项目文件检查")
    print("   2. 依赖环境检查") 
    print("   3. 对比图表生成")
    print("   4. 图表查看功能")
    print("   5. PCE基本使用")
    
    # 执行演示步骤
    steps = [
        check_files,
        check_dependencies,
        generate_charts,
        view_charts,
        demo_pce_usage
    ]
    
    success_count = 0
    for step_func in steps:
        if step_func():
            success_count += 1
        time.sleep(1)  # 短暂暂停
    
    # 总结
    print_header("演示总结")
    print(f"✅ 成功完成: {success_count}/{len(steps)} 个步骤")
    
    if success_count == len(steps):
        print("\n🎉 恭喜！PCE项目演示完全成功！")
        print("\n📖 接下来您可以:")
        print("   • 查看README.md了解详细使用方法")
        print("   • 运行python view_charts.py查看所有对比图表")
        print("   • 使用PCE替代神经网络进行快速推理")
        print("   • 根据需要修改代码适应您的具体问题")
    elif success_count >= 3:
        print("\n✅ 演示基本成功！")
        print("💡 请检查失败的步骤并按照提示解决问题")
    else:
        print("\n❌ 演示遇到问题")
        print("🔧 请检查环境配置和依赖安装")
    
    print(f"\n{'='*60}")
    print("感谢使用PCE神经网络替代方案！")
    print("如有问题，请查看项目文档或提交Issue")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
