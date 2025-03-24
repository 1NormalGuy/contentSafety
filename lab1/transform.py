import os
import re
from PIL import Image
import shutil

# 设置处理训练集和测试集
sets = ['train', 'test']

# 修改为相对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "INRIAPerson")
labels_base_dir = os.path.join(base_dir, "labels")

# 清空并创建标签目录结构
if os.path.exists(labels_base_dir):
    shutil.rmtree(labels_base_dir)
os.makedirs(os.path.join(labels_base_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_base_dir, "val"), exist_ok=True)

# 获取文件夹下所有图片的图片名
def get_name(file_dir):
    list_file = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg' or os.path.splitext(file)[1].lower() == '.png':
                list_file.append(os.path.join(root, file))
    return list_file

# 在labels目录下创建每个图片的标签txt文档
def text_create(name, bnd, dataset_type):
    # 对于train使用train目录，对于test使用val目录
    output_dir = "train" if dataset_type == "train" else "val"
    full_path = os.path.join(labels_base_dir, output_dir, f"{name}.txt")
    
    # 获取对应的图像大小
    image_dir = "Train" if dataset_type == "train" else "Test"
    image_file_path = os.path.join(base_dir, f"INRIAPerson/{image_dir}/pos", f"{name}.png")
    size = get_image_size(image_file_path)
    
    convert_size = convert(size, bnd)
    with open(full_path, 'a') as file:
        file.write(f"0 {convert_size[0]} {convert_size[1]} {convert_size[2]} {convert_size[3]}\n")

# 获取图片的宽高
def get_image_size(image_path):
    im = Image.open(image_path)
    size = im.size
    return size

# 将边界框坐标转换为YOLO格式
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

# 处理各数据集
for dataset_type in sets:
    print(f"处理 {dataset_type} 数据集...")
    
    # 设置标注文件目录
    annotations_dir = os.path.join(base_dir, f"INRIAPerson/{dataset_type.capitalize()}/annotations")
    
    if not os.path.exists(annotations_dir):
        print(f"警告: 标注目录 {annotations_dir} 不存在!")
        continue
        
    annotations = os.listdir(annotations_dir)
    
    for file in annotations:
        if file.endswith('.txt'):
            str_name = file.replace('.txt', '')
            
            if not os.path.isdir(os.path.join(annotations_dir, file)):
                try:
                    with open(os.path.join(annotations_dir, file), encoding='latin1') as f:
                        iter_f = iter(f)
                        for line in iter_f:
                            str_XY = "(Xmax, Ymax)"
                            if str_XY in line:
                                strlist = line.split(str_XY)
                                strlist1 = "".join(strlist[1:])
                                strlist1 = strlist1.replace(':', '')
                                strlist1 = strlist1.replace('-', '')
                                strlist1 = strlist1.replace('(', '')
                                strlist1 = strlist1.replace(')', '')
                                strlist1 = strlist1.replace(',', '')
                                b = strlist1.split()
                                bnd = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                                text_create(str_name, bnd, dataset_type)
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {e}")

print("标签生成完成。训练标签在 labels/train，验证标签在 labels/val。")