# 深度学习对抗攻防实验（Lab 3）

## 项目概述

本实验围绕机器学习模型的安全性展开，重点研究对抗样本的生成、攻击与防御。通过实验，我们将了解深度学习模型面临的安全威胁，以及如何通过对抗训练等技术增强模型鲁棒性。

## 项目结构

```
lab3/mnist-pgd-attack/
├── src/
│   ├── attack/
│   │   ├── __init__.py
│   │   └── pgd.py              # PGD攻击实现
│   ├── models/
│   │   ├── __init__.py
│   │   └── simple_cnn.py       # 简单CNN模型定义
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # 数据加载工具
│   │   └── visualization.py    # 可视化工具
│   ├── train.py                # 模型训练脚本
│   ├── evaluate.py             # 模型评估脚本
│   └── adv_train.py            # 对抗训练脚本
├── data/                       # 数据存储目录
│   └── .gitkeep
├── results/                    # 结果输出目录
│   └── .gitkeep
├── requirements.txt            # 依赖包列表
└── README.md                   # 项目说明文档
```

## 环境配置

### 依赖安装

```bash
# 创建并激活虚拟环境（可选）
conda create -n adv-lab python=3.9
conda activate adv-lab

# 安装依赖
pip install -r requirements.txt
```

主要依赖包：
- PyTorch 1.10.0+
- torchvision 0.11.0+
- numpy
- matplotlib
- tqdm

## 实验内容

本实验包含三个主要部分：

### 1. 对抗样本生成

了解对抗样本的基本概念，以及其对深度学习模型安全性的潜在威胁。

```bash
# 训练基础模型
python src/train.py

# 评估模型在干净样本上的性能
python src/evaluate.py --mode clean
```

### 2. PGD攻击实现

实现并测试投影梯度下降（Projected Gradient Descent, PGD）攻击算法。

```bash
# 生成PGD对抗样本并评估模型鲁棒性
python src/evaluate.py --mode pgd --eps 0.3 --steps 40
```

参数说明：
- `--eps`: 扰动大小上限（L∞范数）
- `--steps`: PGD迭代步数
- `--alpha`: 每步更新大小（默认为eps/10）

### 3. 对抗训练防御

通过将对抗样本加入训练集，提高模型抵抗对抗攻击的能力。

```bash
# 执行对抗训练
python src/adv_train.py --eps 0.2 --steps 10

# 评估对抗训练模型的鲁棒性
python src/evaluate.py --model_path mnist_robust_cnn.pth --mode pgd
```

## 实验结果

### 攻击效果

标准模型在不同强度PGD攻击下的准确率：
- 干净测试集: ~99%
- PGD (eps=0.1, steps=10): ~70%
- PGD (eps=0.2, steps=10): ~40%
- PGD (eps=0.3, steps=40): ~10%

### 防御效果

对抗训练模型在不同强度PGD攻击下的准确率：
- 干净测试集: ~98%
- PGD (eps=0.1, steps=10): ~90%
- PGD (eps=0.2, steps=10): ~85%
- PGD (eps=0.3, steps=40): ~75%

实验结果表明，对抗训练能够显著提高模型对PGD攻击的鲁棒性，但可能会在干净样本上有轻微的准确率下降。

## 可视化结果

实验生成的可视化结果保存在`results/`目录下：
- `pgd_attack_results.png`: 展示原始图像、对抗样本和扰动
- `model_comparison.png`: 展示标准模型和鲁棒模型对同一对抗样本的不同响应
- `accuracy_comparison.png`: 比较两种模型在干净和对抗样本上的准确率

## 实验分析与思考

1. **对抗样本特性**：对抗样本添加的扰动对人眼几乎不可见，但能够成功欺骗深度学习模型。这说明模型的决策边界与人类感知有显著差异。

2. **攻击参数影响**：
   - 扰动大小(eps)增加 → 攻击成功率提高 → 模型准确率下降
   - 迭代步数(steps)增加 → 对抗样本质量提高 → 攻击更有效

3. **对抗训练权衡**：对抗训练能显著提高模型鲁棒性，但通常会导致在干净样本上准确率略有下降，表明鲁棒性和准确性之间存在权衡。

## 参考资料

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.

3. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. IEEE symposium on security and privacy.

## 注意事项

- 对抗样本生成过程可能较为耗时，尤其是在CPU上。建议在GPU环境下运行。
- 参数设置会显著影响攻击效果和训练时间，可根据实际需求调整。
- 对抗训练需要较长时间，可以通过调整`--portion`参数来减少用于生成对抗样本的训练数据比例。