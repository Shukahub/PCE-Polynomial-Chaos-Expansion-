#!/usr/bin/env python3
"""
一键生成所有PCE对比图表
运行此脚本将生成项目中所有的可视化对比图表
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """运行指定的脚本"""
    print(f"\n{'='*60}")
    print(f"正在运行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {description} 完成 (耗时: {end_time-start_time:.1f}秒)")
            return True
        else:
            print(f"❌ {description} 失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"💥 {description} 异常: {e}")
        return False

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    required_packages = [
        'numpy', 'matplotlib', 'sklearn', 'pandas'
    ]
    
    optional_packages = [
        'seaborn', 'memory_profiler', 'psutil'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"   ❌ {package} (必需)")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"   ⚠️  {package} (可选)")
    
    if missing_required:
        print(f"\n❌ 缺少必需依赖: {', '.join(missing_required)}")
        print("请运行: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  缺少可选依赖: {', '.join(missing_optional)}")
        print("建议运行: pip install " + " ".join(missing_optional))
        print("这些依赖用于生成更丰富的图表")
    
    return True

def list_generated_files():
    """列出生成的图表文件"""
    print(f"\n{'='*60}")
    print("📊 生成的图表文件:")
    print(f"{'='*60}")
    
    chart_files = [
        'comprehensive_accuracy_comparison.png',
        'comprehensive_speed_comparison.png', 
        'model_size_comparison.png',
        'memory_usage_comparison.png',
        'deployment_complexity_comparison.png',
        'platform_compatibility_comparison.png',
        'pce_accuracy_analysis.png',
        'pce_training_results.png',
        'pce_vs_nn_comparison.png'
    ]
    
    found_files = []
    missing_files = []
    
    for file in chart_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file:<40} ({size:,} bytes)")
            found_files.append(file)
        else:
            print(f"   ❌ {file:<40} (缺失)")
            missing_files.append(file)
    
    print(f"\n📈 找到 {len(found_files)}/{len(chart_files)} 个图表文件")
    
    if missing_files:
        print(f"⚠️  缺失的文件可能需要运行相应的脚本生成")
    
    return found_files, missing_files

def main():
    """主函数"""
    print("🎨 PCE对比图表生成器")
    print("=" * 60)
    print("此脚本将生成所有PCE vs 神经网络的对比图表")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装缺少的包后重试")
        return
    
    # 要运行的脚本列表
    scripts = [
        ('generate_comparison_charts.py', '综合性能对比图表'),
        ('generate_deployment_charts.py', '部署特性对比图表')
    ]
    
    # 检查脚本文件是否存在
    missing_scripts = []
    for script, _ in scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n❌ 缺少脚本文件: {', '.join(missing_scripts)}")
        return
    
    # 运行所有脚本
    success_count = 0
    total_start_time = time.time()
    
    for script, description in scripts:
        if run_script(script, description):
            success_count += 1
        time.sleep(1)  # 短暂暂停
    
    total_end_time = time.time()
    
    # 列出生成的文件
    found_files, missing_files = list_generated_files()
    
    # 总结
    print(f"\n{'='*60}")
    print("🎯 生成总结")
    print(f"{'='*60}")
    print(f"✅ 成功运行: {success_count}/{len(scripts)} 个脚本")
    print(f"📊 生成图表: {len(found_files)} 个文件")
    print(f"⏱️  总耗时: {total_end_time-total_start_time:.1f} 秒")
    
    if success_count == len(scripts) and len(missing_files) == 0:
        print("\n🎉 所有图表生成完成！")
        print("📖 您现在可以查看README.md中的可视化对比图表")
    elif success_count > 0:
        print("\n✅ 部分图表生成成功")
        print("💡 请检查失败的脚本并重试")
    else:
        print("\n❌ 图表生成失败")
        print("🔧 请检查错误信息并修复问题")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
