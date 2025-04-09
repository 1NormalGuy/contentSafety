import matplotlib.pyplot as plt
import numpy as np

def visualize_results(original_image, adversarial_image, perturbation, original_label, adversarial_label):
    """
    可视化原始图像、对抗样本和扰动

    参数:
        original_image: 原始图像
        adversarial_image: 对抗样本
        perturbation: 扰动
        original_label: 原始图像的标签
        adversarial_label: 对抗样本的标签
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title(f'原始图像\n预测: {original_label}')
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title(f'对抗样本\n预测: {adversarial_label}')
    plt.imshow(adversarial_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title('扰动 (放大 10 倍)')
    plt.imshow(np.clip(perturbation * 10 + 0.5, 0, 1), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()