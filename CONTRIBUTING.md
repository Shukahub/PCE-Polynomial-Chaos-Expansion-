# 贡献指南

感谢您对PCE神经网络替代项目的关注！我们欢迎各种形式的贡献。

## 🚀 快速开始

1. Fork 这个仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📋 贡献类型

### 🐛 Bug 报告
如果您发现了bug，请创建一个issue并包含：
- 详细的问题描述
- 重现步骤
- 期望的行为
- 实际的行为
- 系统环境信息

### 💡 功能请求
对于新功能建议，请包含：
- 功能的详细描述
- 使用场景
- 可能的实现方案
- 对现有功能的影响

### 📚 文档改进
- 修正错别字
- 改进说明的清晰度
- 添加示例
- 翻译文档

### 🔧 代码贡献
- 性能优化
- 新功能实现
- 代码重构
- 测试覆盖率提升

## 🛠️ 开发环境设置

### Python环境
```bash
# 克隆仓库
git clone https://github.com/your-username/pce-neural-network-replacement.git
cd pce-neural-network-replacement

# 安装依赖
pip install -r requirements.txt

# 运行测试
python quick_test.py
```

### Fortran环境
```bash
# Ubuntu/Debian
sudo apt-get install gfortran

# macOS
brew install gcc

# 编译测试
make
```

## 📝 代码规范

### Python代码
- 遵循 PEP 8 风格指南
- 使用有意义的变量名
- 添加适当的注释和文档字符串
- 保持函数简洁（<50行）

### Fortran代码
- 使用现代Fortran语法（Fortran 90+）
- 明确声明变量类型
- 使用适当的缩进（2或4个空格）
- 添加注释说明算法逻辑

### 文档
- 使用Markdown格式
- 包含代码示例
- 保持简洁明了
- 及时更新

## 🧪 测试要求

### 新功能
- 添加相应的单元测试
- 确保所有现有测试通过
- 测试覆盖率不低于80%

### Bug修复
- 添加回归测试
- 验证修复不会引入新问题

### 性能改进
- 提供性能基准测试
- 对比改进前后的性能数据

## 📊 性能基准

在提交性能相关的更改时，请包含：

```python
# 性能测试示例
import time
import numpy as np
from pce_trainer import PCETrainer

def benchmark_inference():
    trainer = PCETrainer()
    trainer.load_model('final_pce_model.pkl')
    
    test_data = np.random.uniform(-1, 1, (1000, 2))
    
    start_time = time.time()
    for _ in range(100):
        predictions = trainer.predict(test_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    throughput = len(test_data) / avg_time
    
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"吞吐量: {throughput:.0f} 样本/秒")

if __name__ == "__main__":
    benchmark_inference()
```

## 🔍 代码审查

所有的Pull Request都会经过代码审查：

### 审查要点
- 代码质量和可读性
- 功能正确性
- 性能影响
- 文档完整性
- 测试覆盖率

### 审查流程
1. 自动化测试通过
2. 至少一位维护者审查
3. 解决所有审查意见
4. 合并到主分支

## 🏷️ 版本发布

### 版本号规则
遵循语义化版本控制 (SemVer)：
- MAJOR.MINOR.PATCH
- MAJOR: 不兼容的API更改
- MINOR: 向后兼容的功能添加
- PATCH: 向后兼容的bug修复

### 发布流程
1. 更新版本号
2. 更新CHANGELOG.md
3. 创建发布标签
4. 发布到GitHub Releases

## 📞 联系方式

如果您有任何问题或建议，请通过以下方式联系：

- 创建GitHub Issue
- 发起GitHub Discussion
- 发送邮件到项目维护者

## 📄 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

---

再次感谢您的贡献！🎉
