import torch
import torch.nn as nn

class PGDAttack:
    def __init__(self, model, eps=0.3, alpha=0.01, steps=40, random_start=True):
        """
        PGD 攻击实现 (针对 MNIST)
        
        参数:
            model: 目标模型
            eps: 最大扰动大小 (L∞ 范数)
            alpha: 每一步的步长
            steps: 迭代步数
            random_start: 是否从随机点开始
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss_fn = nn.CrossEntropyLoss()
        
    def attack(self, images, labels):
        """
        对输入图像生成对抗样本
        
        参数:
            images: 输入图像批次
            labels: 目标标签
            
        返回:
            对抗样本
        """
        images = images.clone().detach().to(images.device)
        labels = labels.clone().detach().to(labels.device)
        
        adv_images = images.clone().detach()
        
        # 随机初始化
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=-1.0, max=1.0)  # MNIST 归一化范围为 [-1, 1]
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            
            # 前向传播
            outputs = self.model(adv_images)
            loss = self.loss_fn(outputs, labels)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 获取梯度
            grad = adv_images.grad.data
            
            # FGSM 步骤
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # 投影步骤 - 确保扰动在 eps 球内
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1.0, max=1.0)  # MNIST 归一化范围为 [-1, 1]
            
        return adv_images