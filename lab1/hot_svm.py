import numpy as np
import time
import random
import os
import pickle
import joblib
from tqdm import tqdm
from cv2 import rectangle, imshow, waitKey
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def clip_image(img, left, top,
    width=64, height=128):
    '''
    截取图片的一个区域。

    参数
    ---
    img: 图片输入。
    left: 区域左边的坐标。
    top: 区域上边的坐标。
    width: 区域宽度。
    height: 区域高度。
    '''
    return img[top:top + height, left:left + width]

def extract_hog_feature(img):
    '''
    提取单个图像img的HOG特征。
    '''
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    ).astype('float32')

def read_images(pos_dir, neg_dir,
    neg_area_count, description):
    '''
    读取图片，提取样本HOG特征。

    参数
    ---
    pos_dir: 正样本所在文件夹。
    neg_dir: 负样本所在文件夹。
    neg_area_count: 在每个负样本中随机截取区域的个数。
    description: 用途描述（训练/测试）。

    返回值
    -----
    返回一个元组(x, y)，x是所有图片的HOG特征，
    y是所有图片的分类（1=正样本，0=负样本）。
    '''
    pos_img_files = os.listdir(pos_dir)
    # 正样本文件列表
    neg_img_files = os.listdir(neg_dir)
    # 负样本文件列表

    area_width = 64 # 截取的区域宽度
    area_height = 128 # 截取的区域高度

    x = [] # 图片的HOG特征
    y = [] # 图片的分类

    for pos_file in tqdm(pos_img_files,
        desc=f'{description}正样本'):
        # 读取所有正样本
        pos_path = os.path.join(pos_dir, pos_file)
        # 正样本路径
        pos_img = imread(pos_path, as_gray=True)
        # 正样本图片
        img_height, img_width = pos_img.shape
        # 该图片的宽、高
        clip_left = (img_width - area_width) // 2
        # 截取区域的左边
        clip_top = (img_height - area_height) // 2
        # 截取区域的上边
        pos_center = clip_image(pos_img,
            clip_left, clip_top, area_width, area_height)
        # 截取中间部分
        hog_feature = extract_hog_feature(
            pos_center) # 提取HOG特征
        x.append(hog_feature) # 加入HOG向量
        y.append(1) # 1代表正类
    
    for neg_file in tqdm(neg_img_files,
        desc=f'{description}训练负样本'):
        # 读取所有负样本
        neg_path = os.path.join(neg_dir, neg_file)
        # 负样本路径
        neg_img = imread(neg_path, as_gray=True)
        # 负样本图片
        img_height, img_width = neg_img.shape
        # 该图片的宽、高
        left_max = img_width - area_width
        # 区域左边坐标的最大值
        top_max = img_height - area_height
        # 区域
        for _ in range(neg_area_count):
            # 随机截取neg_area_count个区域
            left = random.randint(0, left_max) # 区域左边
            top = random.randint(0, top_max) # 区域上边
            clipped_area = clip_image(neg_img,
                left, top, area_width, area_height)
            # 截取的区域
            hog_feature = extract_hog_feature(
                clipped_area) # 提取HOG特征
            x.append(hog_feature)
            y.append(0)
    return x, y

def read_training_data():
    '''
    读取训练数据。
    '''
    pos_dir = 'INRIAPerson/96X160H96/Train/pos'
    neg_dir = 'INRIAPerson/Train/neg'
    neg_area_count = 10
    description = '训练'
    return read_images(pos_dir, neg_dir,
        neg_area_count, description)

def read_test_data():
    '''
    读取测试数据。
    '''
    pos_dir = 'INRIAPerson/70X134H96/Test/pos'
    neg_dir = 'INRIAPerson/Test/neg'
    neg_area_count = 10
    description = '测试'
    return read_images(pos_dir, neg_dir,
        neg_area_count, description)

def save_hog(x, y, filename):
    '''
    把read_training_samples的返回值（x, y）
    写入名为filename的文件。
    '''
    with open(filename, 'wb') as file:
        pickle.dump((x, y), file)

def load_hog(filename):
    '''
    从名为filename的文件中加载训练数据（x, y）。
    '''
    result = None
    with open(filename, 'rb') as file:
        result = pickle.load(file)
    return result

def train_SVM(x, y):
    '''
    训练SVM。

    参数
    ---
    x, y: read_training_samples的返回值。

    返回值
    -----
    返回训练所得的SVM。
    '''
    SVM = SVC(
        tol=1e-6,
        C=0.01,
        max_iter=-1,
        gamma='auto',
        kernel='rbf',
        probability=True
    ) # 创建SVM实例
    SVM.fit(x, y) # 进行训练
    return SVM

def test_SVM(SVM, test_data, show_stats=False):
    '''
    测试训练好的SVM。

    参数
    ---
    SVM: 训练好的SVM模型。
    test_data: 测试数据（read_test_data的返回值）。
    show_stats: 是否显示统计数据（miss rate vs.
    false positive rate曲线）。

    返回值
    -----
    返回AUC（ROC曲线下的面积）。AUC介于0.5和1之间。
    AUC越接近1，模型越可靠。
    '''
    hog_features = test_data[0] # 测试数据的HOG特征
    labels = np.array(test_data[1]) # 确保标签是NumPy数组
    prob = SVM.predict_proba(hog_features)[:, 1]
    if show_stats:
        # 下面将prob和labels按prob的降序排序
        sorted_indices = np.argsort(
            prob, kind="mergesort")[::-1]
        labels = labels[sorted_indices]
        prob = prob[sorted_indices]
        distinct_value_indices = np.where(np.diff(prob))[0]
            # prob中不同值第一次出现的下标
        threshold_idxs = np.r_[
            distinct_value_indices, labels.size - 1]
            # 阈值的下标，在末尾增加了最后一个样本的下标
        tps = np.cumsum(labels)[threshold_idxs]
            # 不同概率阈值对应的真正例数。
            # 注意现在已经按prob的降序排序，
            # 这种写法正确的原因是：在数组某一位置前的概率
            # 一定大于阈值，在此之后的概率一定小于阈值，
            # 所以真正例数就是在这一位置之前的正样本数。
        fps = 1 + threshold_idxs - tps
            # 不同概率阈值对应的假正例数。
            # threshold_idxs存储的是下标，
            # 加一后变成个数，
            # 再减去真正例数就是假正例数。
        num_positive = tps[-1]
            # tps的最后一项就是labels的和，
            # 因此代表正例的个数。
        recall = tps / num_positive
            # 查全率就是在所有正例中查出了多少真正例。
        miss = 1 - recall # 计算miss
        num_negative = fps[-1] # 负例个数
        fpr = fps / num_negative
            # 假阳性率（false positive rate）
        plt.plot(miss, fpr, color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Miss Rate')
        plt.title('Miss Rate - '
            'False Positive Rate Curve')
        plt.show()
    AUC = metrics.roc_auc_score(labels, prob)
    return AUC

def area_of_box(box):
    '''
    计算框的面积。

    参数
    ---
    box: 框，格式为(left, top, width, height)。

    返回值
    -----
    box的面积，即width * height。
    '''
    return box[2] * box[3]

def intersection_over_union(box1, box2):
    '''
    两个框的交并比（IoU）。

    参数
    ---
    box1: 边框1。
    box2: 边框2。
    '''
    intersection_width = max(0,
        box1[0] + box1[2] - box2[0])
        # 相交部分宽度=max(0, box1的右边 - box2的左边)
    intersection_height = max(0,
        box1[1] + box1[3] - box2[1])
        # 相交部分长度=max(0, box1的下边 - box2的上边)
    intersection_area = intersection_width * \
        intersection_height # 相交部分面积
    area_box1 = area_of_box(box1) # box1的面积
    area_box2 = area_of_box(box2) # box1的面积
    union_area = area_box1 + area_box2 - \
        intersection_area
    if abs(union_area) < 1:
        IoU = 0 # 防止除以0
    else:
        IoU = intersection_area / union_area
            # 并集的面积等于二者面积之和减去交集的面积
    return IoU

def non_maximum_suppression(pos_box_list, pos_prob,
    IoU_threshold=0.4):
    '''
    非极大值抑制（NMS）。

    参数
    ---
    pos_box_list: 含有人的概率大于阈值的边框列表。
    pos_prob: 对应的概率。
    IoU_threshold: 舍弃边框的IoU阈值。

    返回值
    -----
    抑制后的边框列表。
    '''
    result = [] # 结果
    for box1, prob1 in zip(pos_box_list, pos_prob):
        discard = False # 是否舍弃box1
        for box2, prob2 in zip(
            pos_box_list, pos_prob):
            if intersection_over_union(
                box1, box2) > IoU_threshold:
                # IoU大于阈值
                if prob2 > prob1: # 舍弃置信度较小的
                    discard = True
                    break
        if not discard: # 未舍弃box1
            result.append(box1) # 加入结果列表
    return result

def detect_pedestrian(SVM, filename, show_img=False,
    threshold=0.99, area_width=64, area_height=128,
    min_width=48, width_scale=1.25, coord_step=16,
    ratio=2):
    '''
    用SVM检测file文件中的行人，采用非极大值抑制（NMS）
    避免重复画框。

    参数
    ---
    SVM: 训练好的SVM模型。
    filename: 输入文件名。
    show_img: 是否给用户显示已画框的图片。
    threshold: 将某一部分视为人的概率阈值。
    area_width: 缩放后区域的宽度。
    area_height: 缩放后区域的高度。
    min_width: 框宽度的最小值，也是初始值。
    width_scale: 每一次框宽度增大时扩大的倍数。
    coord_step: 坐标变化的步长。
    ratio: 框的长宽比。

    返回值
    -----
    一个列表，每个列表项是一个元组
    (left, top, width, height), 为行人的边框。
    '''
    box_list = [] # 行人边框列表
    hog_list = [] # HOG特征列表
    with open(filename, 'rb') as file:
        img = imread(file, as_gray=True) # 读取文件
        img_height, img_width = img.shape # 图片长宽
        width = min_width # 框的宽度
        height = int(width * ratio) # 框的长度
        while width < img_width and height < img_height:
            for left in range(0, img_width - width,
                coord_step): # 框的左侧
                for top in range(0, img_height - height,
                    coord_step): # 框的上侧
                    patch = clip_image(img, left, top,
                        width, height) # 截取图像的一部分
                    resized = resize(patch,
                        (area_height, area_width))
                        # 缩放图片
                    hog_feature = extract_hog_feature(
                        resized) # 提取HOG特征
                    box_list.append((left, top,
                        width, height))
                    hog_list.append(hog_feature)
            width = int(width * width_scale)
            height = width * ratio
        prob = SVM.predict_proba(hog_list)[:, 1]
            # 用SVM模型进行判断
        mask = (prob >= threshold)
            # 布尔数组, mask[i]代表prob[i]是否等于阈值

        box_list_np = np.array(box_list)
        if len(mask) > 0 and len(box_list_np) > 0:
            pos_box_list = box_list_np[mask]
                # 含有人的框
            pos_prob = prob[mask]  # 对应的预测概率
            box_list_after_NMS = non_maximum_suppression(
                pos_box_list, pos_prob)
                # NMS处理之后的框列表


        box_list_after_NMS = non_maximum_suppression(
            pos_box_list, pos_prob)
            # NMS处理之后的框列表
        if show_img:
            shown_img = np.array(img)
                # 复制原图像，准备画框
            for box in box_list_after_NMS:
                shown_img = rectangle(shown_img,
                    pt1=(box[0], box[1]),
                    pt2=(box[0] + box[2],
                        box[1] + box[3]),
                    color=(0, 0, 0),
                    thickness=2)
            imshow('', shown_img)
            waitKey(0)
        return box_list_after_NMS

def detect_multiple_images(SVM, dir):
    '''
    检测多个图像文件（dir文件夹中所有文件）中的行人。

    参数
    ---
    SVM: 训练好的SVM模型。
    dir: 存放图片的文件夹。
    '''
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        detect_pedestrian(SVM, file_path,
            show_img=True)

if __name__ == '__main__':
    print('execution starts')

    random.seed(time.time()) # 设置随机数种子
    x, y = read_training_data() # 读取训练数据，提取HOG特征
    save_hog(x, y, 'hog_xy.pickle')
    print('training data hog extraction done')

    test_data = read_test_data() # 读取测试数据，提取HOG特征
    save_hog(*test_data, 'test_data_hog.pickle')
    print('test data hog extraction done')

    x, y = load_hog('hog_xy.pickle') # 训练SVM模型
    time_before_training = time.time()
    SVM = train_SVM(x, y)
    time_after_training = time.time()
    print('SVM training done, cost %.2fs.' % \
        (time_after_training - time_before_training))
    joblib.dump(SVM, 'SVM.model', compress=9)

    SVM = joblib.load('SVM.model') # 测试SVM模型
    test_data = load_hog('test_data_hog.pickle')
    print('AUC=%.8f.' % test_SVM(SVM, test_data, True))

    detect_multiple_images(SVM, # 用SVM模型识别图片
        'INRIAPerson/Test/pos')
