#!/usr/bin/env python3
"""
PCE对比图表查看器
快速预览所有生成的对比图表
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def view_all_charts():
    """显示所有对比图表"""
    
    # 图表文件列表
    chart_files = [
        ('comprehensive_accuracy_comparison.png', '综合精度对比'),
        ('comprehensive_speed_comparison.png', '综合速度对比'),
        ('model_size_comparison.png', '模型大小对比'),
        ('memory_usage_comparison.png', '内存使用对比'),
        ('deployment_complexity_comparison.png', '部署复杂度对比'),
        ('platform_compatibility_comparison.png', '平台兼容性对比'),
        ('pce_accuracy_analysis.png', '精度分析'),
        ('pce_training_results.png', '训练结果'),
        ('pce_vs_nn_comparison.png', 'PCE vs NN对比')
    ]
    
    # 检查文件是否存在
    existing_charts = []
    missing_charts = []
    
    for filename, title in chart_files:
        if os.path.exists(filename):
            existing_charts.append((filename, title))
        else:
            missing_charts.append((filename, title))
    
    if not existing_charts:
        print("❌ 未找到任何图表文件")
        print("请先运行 python generate_all_charts.py 生成图表")
        return
    
    print(f"📊 找到 {len(existing_charts)} 个图表文件")
    
    if missing_charts:
        print(f"⚠️  缺失 {len(missing_charts)} 个图表文件:")
        for filename, title in missing_charts:
            print(f"   - {title} ({filename})")
    
    # 显示图表
    print("\n🖼️  正在显示图表...")
    
    # 计算网格布局
    n_charts = len(existing_charts)
    if n_charts <= 4:
        rows, cols = 2, 2
    elif n_charts <= 6:
        rows, cols = 2, 3
    elif n_charts <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('PCE vs 神经网络 - 对比图表总览', fontsize=16, fontweight='bold')
    
    # 显示每个图表
    for i, (filename, title) in enumerate(existing_charts):
        if i >= rows * cols:
            break
            
        try:
            # 读取图片
            img = mpimg.imread(filename)
            
            # 创建子图
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            print(f"⚠️  无法显示 {filename}: {e}")
    
    plt.tight_layout()
    plt.show()

def view_single_chart(chart_name):
    """显示单个图表"""
    
    chart_files = {
        'accuracy': 'comprehensive_accuracy_comparison.png',
        'speed': 'comprehensive_speed_comparison.png',
        'size': 'model_size_comparison.png',
        'memory': 'memory_usage_comparison.png',
        'deployment': 'deployment_complexity_comparison.png',
        'platform': 'platform_compatibility_comparison.png',
        'analysis': 'pce_accuracy_analysis.png',
        'training': 'pce_training_results.png',
        'comparison': 'pce_vs_nn_comparison.png'
    }
    
    if chart_name not in chart_files:
        print(f"❌ 未知的图表名称: {chart_name}")
        print("可用的图表:")
        for key, filename in chart_files.items():
            print(f"   {key}: {filename}")
        return
    
    filename = chart_files[chart_name]
    
    if not os.path.exists(filename):
        print(f"❌ 图表文件不存在: {filename}")
        return
    
    try:
        # 读取并显示图片
        img = mpimg.imread(filename)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.title(f'PCE对比图表: {filename}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ 显示图表: {filename}")
        
    except Exception as e:
        print(f"❌ 无法显示图表: {e}")

def list_charts():
    """列出所有可用的图表"""
    
    chart_files = [
        ('comprehensive_accuracy_comparison.png', '综合精度对比'),
        ('comprehensive_speed_comparison.png', '综合速度对比'),
        ('model_size_comparison.png', '模型大小对比'),
        ('memory_usage_comparison.png', '内存使用对比'),
        ('deployment_complexity_comparison.png', '部署复杂度对比'),
        ('platform_compatibility_comparison.png', '平台兼容性对比'),
        ('pce_accuracy_analysis.png', '精度分析'),
        ('pce_training_results.png', '训练结果'),
        ('pce_vs_nn_comparison.png', 'PCE vs NN对比')
    ]
    
    print("📊 PCE对比图表列表:")
    print("=" * 60)
    
    for i, (filename, title) in enumerate(chart_files, 1):
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            status = f"✅ ({size:,} bytes)"
        else:
            status = "❌ (缺失)"
        
        print(f"{i:2d}. {title:<25} {filename:<35} {status}")
    
    print("=" * 60)

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) == 1:
        # 无参数，显示所有图表
        print("🎨 PCE对比图表查看器")
        print("=" * 40)
        view_all_charts()
        
    elif len(sys.argv) == 2:
        command = sys.argv[1].lower()
        
        if command == 'list':
            list_charts()
        elif command == 'all':
            view_all_charts()
        elif command in ['help', '-h', '--help']:
            print("🎨 PCE对比图表查看器")
            print("=" * 40)
            print("用法:")
            print("  python view_charts.py           # 显示所有图表")
            print("  python view_charts.py all       # 显示所有图表")
            print("  python view_charts.py list      # 列出图表文件")
            print("  python view_charts.py accuracy  # 显示精度对比图")
            print("  python view_charts.py speed     # 显示速度对比图")
            print("  python view_charts.py size      # 显示大小对比图")
            print("  python view_charts.py memory    # 显示内存对比图")
            print("  python view_charts.py deployment # 显示部署对比图")
            print("  python view_charts.py platform  # 显示平台对比图")
            print("  python view_charts.py analysis  # 显示精度分析图")
            print("  python view_charts.py training  # 显示训练结果图")
            print("  python view_charts.py comparison # 显示对比图")
        else:
            view_single_chart(command)
    else:
        print("❌ 参数过多")
        print("使用 python view_charts.py help 查看帮助")

if __name__ == "__main__":
    main()
