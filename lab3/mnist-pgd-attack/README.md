# MNIST PGD 攻击项目

本项目展示了对在 MNIST 数据集上训练的简单卷积神经网络（CNN）模型实施投影梯度下降（PGD）攻击的实现。项目目标是展示如何执行和评估对抗性攻击。

## 项目结构

```
mnist-pgd-attack
├── src
│   ├── attack
│   │   ├── __init__.py
│   │   └── pgd.py
│   ├── models
│   │   ├── __init__.py
│   │   └── simple_cnn.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── visualization.py
│   ├── train.py
│   └── evaluate.py
├── data
│   └── .gitkeep
├── results
│   └── .gitkeep
├── requirements.txt
├── README.md
└── .gitignore
```

## 安装

要设置项目，请克隆仓库并安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. **数据准备**：当您运行训练脚本时，MNIST 数据集将自动下载。

2. **训练模型**：您可以使用以下命令训练 CNN 模型：

   ```bash
   python src/train.py
   ```

3. **评估模型**：训练后，评估模型性能并生成对抗样本：

   ```bash
   python src/evaluate.py
   ```

4. **运行主脚本**：要执行整个过程（训练、评估和 PGD 攻击），运行：

   ```bash
   python src/main.py
   ```

## 结果

攻击结果和模型性能将保存在 `results` 目录中。您可以使用提供的可视化工具查看原始图像、对抗样本和扰动。

## 许可证

本项目基于 MIT 许可证。有关详细信息，请参阅 LICENSE 文件。

## 致谢

- MNIST 数据集由 Yann LeCun、Corinna Cortes 和 Christopher J.C. Burges 提供。
- PGD 攻击的实现受到对抗性机器学习领域多项研究论文的启发。