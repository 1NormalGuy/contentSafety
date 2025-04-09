import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from models.simple_cnn import SimpleCNN
from attack.pgd import PGDAttack

def generate_adversarial_examples(model, data_loader, device, eps=0.2, steps=10, portion=0.5):
    """
    为训练集生成对抗样本
    
    参数:
        model: 目标模型
        data_loader: 数据加载器
        device: 使用设备
        eps: 扰动大小
        steps: PGD 步数
        portion: 要生成对抗样本的数据比例 (0-1)
        
    返回:
        包含对抗样本和标签的数据集
    """
    pgd_attack = PGDAttack(model, eps=eps, steps=steps)
    model.eval()
    
    adv_images_list = []
    labels_list = []
    
    print(f"生成对抗样本，扰动大小: {eps}, 步数: {steps}")
    start_time = time.time()
    
    total_batches = len(data_loader)
    process_batches = int(total_batches * portion)
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx >= process_batches:
            break
            
        if batch_idx % 10 == 0:
            print(f"处理批次 {batch_idx}/{process_batches}...")
            
        images, labels = images.to(device), labels.to(device)
        
        # 生成对抗样本
        adv_images = pgd_attack.attack(images, labels)
        
        adv_images_list.append(adv_images.detach().cpu())
        labels_list.append(labels.cpu())
    
    adv_images = torch.cat(adv_images_list)
    labels = torch.cat(labels_list)
    
    print(f"对抗样本生成完成。用时: {time.time() - start_time:.2f}秒")
    print(f"生成的对抗样本数量: {len(labels)}")
    
    return TensorDataset(adv_images, labels)

def train_with_adversarial_examples(model, train_loader, adv_dataset, val_loader, device, epochs=5):
    """
    使用原始数据和对抗样本进行混合训练
    
    参数:
        model: 要训练的模型
        train_loader: 原始训练数据加载器
        adv_dataset: 对抗样本数据集
        val_loader: 验证数据加载器
        device: 使用设备
        epochs: 训练轮数
    """
    # 直接创建新的数据加载器，不使用 ConcatDataset，避免数据类型不匹配问题
    adv_loader = DataLoader(adv_dataset, batch_size=train_loader.batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0
    
    print(f"\n开始对抗训练，原始数据量: {len(train_loader.dataset)}, 对抗样本数量: {len(adv_dataset)}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        # 首先训练原始数据
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 99:
                print(f'[Epoch {epoch + 1}, 原始数据 Batch {batch_idx + 1}] loss: {running_loss / batch_count:.3f}')
        
        # 然后训练对抗样本
        for batch_idx, (inputs, targets) in enumerate(adv_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 99:
                print(f'[Epoch {epoch + 1}, 对抗样本 Batch {batch_idx + 1}] loss: {running_loss / batch_count:.3f}')
        
        print(f'[Epoch {epoch + 1}] 平均损失: {running_loss / batch_count:.3f}')
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, 验证准确率: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'mnist_robust_cnn.pth')
            print(f'模型已保存，准确率: {best_accuracy:.2f}%')
    
    return model

def evaluate_model_robustness(standard_model, robust_model, test_loader, device):
    """
    评估标准模型和对抗训练模型的鲁棒性
    
    参数:
        standard_model: 标准训练模型
        robust_model: 对抗训练模型
        test_loader: 测试数据加载器
        device: 使用设备
    """
    for model_name, model in [("标准模型", standard_model), ("鲁棒模型", robust_model)]:
        model.eval()
        
        # 在干净测试集上评估
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        clean_accuracy = 100 * correct / total
        print(f"{model_name} 在干净测试集上的准确率: {clean_accuracy:.2f}%")
        
        # 在不同扰动强度的PGD攻击下评估
        for eps in [0.1, 0.2, 0.3]:
            for steps in [10, 40]:
                pgd_attack = PGDAttack(model, eps=eps, steps=steps)
                correct = 0
                total = 0
                
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # 生成对抗样本
                    adv_images = pgd_attack.attack(images, labels)
                    
                    # 预测
                    with torch.no_grad():
                        outputs = model(adv_images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    if total >= 1000:  # 只测试部分样本以节省时间
                        break
                
                adv_accuracy = 100 * correct / total
                print(f"{model_name} 在 PGD 攻击下的准确率 (eps={eps}, steps={steps}): {adv_accuracy:.2f}%")

def visualize_comparative_results(standard_model, robust_model, test_loader, device, eps=0.3, steps=40):
    """
    可视化比较标准模型和对抗训练模型的表现
    """
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # 为两个模型创建PGD攻击器
    standard_attack = PGDAttack(standard_model, eps=eps, steps=steps)
    robust_attack = PGDAttack(robust_model, eps=eps, steps=steps)
    
    # 为标准模型生成对抗样本
    adv_images_standard = standard_attack.attack(images, labels)
    
    # 为鲁棒模型生成对抗样本
    adv_images_robust = robust_attack.attack(images, labels)
    
    # 得到预测结果
    with torch.no_grad():
        # 标准模型
        standard_clean_outputs = standard_model(images)
        standard_adv_outputs = standard_model(adv_images_standard)
        standard_clean_preds = torch.argmax(standard_clean_outputs, dim=1)
        standard_adv_preds = torch.argmax(standard_adv_outputs, dim=1)
        
        # 鲁棒模型
        robust_clean_outputs = robust_model(images)
        robust_adv_outputs = robust_model(adv_images_standard)  # 使用标准模型的对抗样本测试鲁棒模型
        robust_clean_preds = torch.argmax(robust_clean_outputs, dim=1)
        robust_adv_preds = torch.argmax(robust_adv_outputs, dim=1)
    
    def denormalize(x):
        return x * 0.5 + 0.5
    
    # 可视化结果
    plt.figure(figsize=(15, 12))
    
    for i in range(5):
        # 原始图像
        plt.subplot(5, 3, i*3+1)
        plt.imshow(denormalize(images[i][0].cpu()), cmap='gray')
        plt.title(f"Original: {labels[i].item()}")
        plt.axis('off')
        
        # 标准模型对抗样本
        plt.subplot(5, 3, i*3+2)
        plt.imshow(denormalize(adv_images_standard[i][0].cpu().detach()), cmap='gray')
        plt.title(f"Standard Model: {standard_adv_preds[i].item()}")
        plt.axis('off')
        
        # 鲁棒模型对同一对抗样本的预测
        plt.subplot(5, 3, i*3+3)
        plt.imshow(denormalize(adv_images_standard[i][0].cpu().detach()), cmap='gray')
        plt.title(f"Robust Model: {robust_adv_preds[i].item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    # 生成结果摘要
    standard_clean_acc = (standard_clean_preds == labels).float().mean().item() * 100
    standard_adv_acc = (standard_adv_preds == labels).float().mean().item() * 100
    robust_clean_acc = (robust_clean_preds == labels).float().mean().item() * 100
    robust_adv_acc = (robust_adv_preds == labels).float().mean().item() * 100
    
    # 绘制比较图表
    models = ['Standard Model', 'Robust Model']
    clean_acc = [standard_clean_acc, robust_clean_acc]
    adv_acc = [standard_adv_acc, robust_adv_acc]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, clean_acc, width, label='Clean Accuracy')
    rects2 = ax.bar(x + width/2, adv_acc, width, label='Adversarial Accuracy')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.1f')
    ax.bar_label(rects2, fmt='%.1f')
    
    fig.tight_layout()
    plt.savefig('results/accuracy_comparison.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("加载 MNIST 数据集...")
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # 分割训练集为训练和验证
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("准备标准模型...")
    standard_model = SimpleCNN().to(device)
    
    # 训练或加载标准模型
    try:
        standard_model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("成功加载预训练的标准模型")
    except FileNotFoundError:
        print("没有找到预训练模型，正在训练标准模型...")
        
        # 训练标准模型
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(standard_model.parameters(), lr=0.001)
        
        for epoch in range(5):
            standard_model.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = standard_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            
            # 验证
            standard_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = standard_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch {epoch + 1}, 验证准确率: {accuracy:.2f}%')
        
        torch.save(standard_model.state_dict(), 'mnist_cnn.pth')
        print("标准模型训练完成并保存")
    
    # 使用标准模型生成对抗样本
    print("\n生成对抗样本用于对抗训练...")
    adv_dataset = generate_adversarial_examples(standard_model, train_loader, device, eps=0.2, steps=10, portion=0.5)
    
    # 创建鲁棒模型
    print("\n创建鲁棒模型...")
    robust_model = SimpleCNN().to(device)
    
    # 进行对抗训练
    print("\n使用对抗样本进行对抗训练...")
    train_with_adversarial_examples(robust_model, train_loader, adv_dataset, val_loader, device, epochs=5)
    
    # 加载最佳鲁棒模型
    robust_model.load_state_dict(torch.load('mnist_robust_cnn.pth'))
    
    # 评估模型鲁棒性
    print("\n评估模型鲁棒性...")
    evaluate_model_robustness(standard_model, robust_model, test_loader, device)
    
    # 可视化比较结果
    print("\n生成可视化比较结果...")
    visualize_comparative_results(standard_model, robust_model, test_loader, device)
    
    print("\n实验完成! 结果保存在 'results/' 目录")

if __name__ == "__main__":
    main()