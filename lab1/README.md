# 行人检测项目
武汉大学国家网络安全学院2024-2025学年内容安全实验一

## 项目概述

本项目利用机器学习和深度学习方法实现了行人检测功能，包含两种实现方式：
1. 传统机器学习方法：HOG特征提取 + SVM分类器
2. 深度学习方法：YOLOv8目标检测模型

项目使用INRIA Person数据集进行训练和测试，提供了从数据预处理、特征提取、模型训练到模型评估的完整工作流程。

## 环境设置

### 依赖库
```bash
# 创建并激活虚拟环境
conda create -n yolo8 python=3.9
conda activate yolo8

# 安装依赖
pip install numpy scikit-learn scikit-image pillow torch ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据集准备

本项目使用INRIA Person数据集，目录结构如下：
```
INRIAPerson/
├── Train/
│   ├── pos/          
│   └── annotations/   
├── Test/
│   ├── pos/          
│   └── annotations/   
└── Label/
    ├── train/           
    └── val/   
```

## 项目结构

```
contentSafety/lab1/
├── INRIAPerson/       # INRIA Person数据集
├── hot_svm.py         # HOG+SVM实现代码
├── transform.py       # 将INRIA标注转换为YOLO格式的脚本
├── labels/            # 转换后的YOLO格式标签
│   ├── train/         # 训练集标签
│   └── val/           # 验证集标签
└── ultralytics/       # YOLOv8相关代码
    ├── train_yolov8.py           # YOLOv8训练脚本
    └── ultralytics/cfg/datasets/inriaperson.yaml  # 数据集配置文件
```

## 使用指南

### HOG+SVM方法

1. 运行HOG+SVM训练和测试：
```bash
python hot_svm.py
```

这将执行以下步骤：
- 从INRIA数据集提取HOG特征
- 训练SVM模型
- 评估SVM模型性能
- 使用训练好的模型在测试图像上进行行人检测

### YOLOv8方法

1. 转换数据集标注为YOLO格式：
```bash
python transform.py
```

2. 训练YOLOv8模型：
```bash
cd ultralytics
python train_yolov8.py
```

3. 训练结果将保存在 detect 目录中

## 实现详情

### HOG+SVM实现

hot_svm.py 实现了以下功能：
- HOG特征提取
- SVM模型训练
- 使用非极大值抑制(NMS)合并重叠检测框
- 在测试图像上进行行人检测和可视化

### YOLOv8实现

- 使用 transform.py 将INRIA标注转换为YOLO兼容格式
- 通过 inriaperson.yaml 配置数据集路径和类别信息
- 使用预训练的YOLOv8模型进行迁移学习
- 支持验证评估和结果可视化

## TIPS

1. 图像路径配置：确保YAML配置文件中的路径与实际数据集路径一致
2. 标签转换：运行YOLOv8前必须先执行 transform.py 生成标签文件
3. PNG警告：训练过程中的 "libpng warning: iCCP: known incorrect sRGB profile" 是图像格式警告，不影响训练
4. 确保标签文件与图像文件对应，且格式正确
5. 部分代码文件可能在非macos上出现报错，均为正常现象


## 参考资料

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [INRIA Person 数据集](http://pascal.inrialpes.fr/data/human/)
- [HOG特征描述符论文](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

---

author:luinage