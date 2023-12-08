import os
dll_directory = r'E:\XinYuTan\pathlogicl-main\openslide-win64-20221217\bin'
os.add_dll_directory(dll_directory)
print("all thing is ok")
import openslide
import json

import numpy as np
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision import transforms
import geopandas
from shapely.geometry import Point

classification_name = 'None'  # classification_name是全局变量，根据is_inside_annotation_center的判断给对应的patch的赋值label
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载ResNet-50预训练模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置为评估模式，以便不影响Batch Normalization层的统计信息

# 转换输入图像的大小和格式
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 设定数据集
wsi_folder = 'data/wsi'
annotation_folder = 'data/annotation'

def is_background(start_x, start_y, level, patch_size, slide):
    # 定义背景颜色阈值
    background_threshold = 0.8  # 如果切片中超过80%的像素接近背景颜色，则不保存
    background_color_lower = [200, 200, 200, 0]  # 下界，包括了RGBA通道
    background_color_upper = [255, 255, 255, 255]  # 上界，包括了RGBA通道
    region = slide.read_region((start_x, start_y), level, patch_size)
    region_array = np.array(region)
    within_background = np.all((region_array >= background_color_lower) & (region_array <= background_color_upper),
                               axis=-1)
    background_pixels = np.sum(within_background) / np.prod(region_array.shape[:2])
    # 判断是否保存切片，如果背景像素比例低于阈值则保存
    return background_pixels > background_threshold

def is_inside_annotation_center(patch_box, geojson_data):
    global classification_name  # 使用global关键字以确保更新全局变量
    classification_name = 'None'  # classification_name是全局变量，根据is_inside_annotation_center的判断给对应的patch的赋值label

    # 矩形中心坐标 (x, y)
    x, y = (patch_box[0] + patch_box[2]) / 2, (patch_box[1] + patch_box[3]) / 2
    # 创建一个点对象表示矩形中心
    point = Point(x, y)

    for index, row in geojson_data.iterrows():
        if row['geometry'].contains(point):
            print(f"Rectangle is inside: geometry {index}，its classification is :{geojson_data['classification'][index]['name']}")
            classification_name = geojson_data['classification'][index]['name']
            return True

    return False

def extract_features_from_patch(patch, model):
    patch = patch.convert('RGB')
    patch_tensor = transform(patch).unsqueeze(0)

    with torch.no_grad():
        features = model(patch_tensor)

    return features.squeeze()

class_labels = {
    'basic cell': 0,
    'shadow cell': 1,
    'Other': 2,
    'None': 3
}

# 获取wsi文件夹中所有bif和svs文件的路径
wsi_files = [f for f in os.listdir(wsi_folder) if f.endswith(('.bif', '.svs'))]

for wsi_file in wsi_files:
    wsi_path = os.path.join(wsi_folder, wsi_file)
    annotation_file = os.path.join(annotation_folder, os.path.splitext(wsi_file)[0] + '.geojson')

    # 打开病理图像文件并创建幻灯片对象
    with openslide.OpenSlide(wsi_path) as slide:
        # 病理文件对应的标注数据
        with open(annotation_file, 'r') as geojson_file:
            annotation_data = geopandas.read_file(geojson_file)

        resolutions = [0, 1, 2, 3]

        for resolution_level in resolutions:
            scale = 1.0 / (2 ** resolution_level)
            patch_size = (224, 224)

            folder_name = f'window_images_resolution_{resolution_level}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            extracted_coordinates = []
            skipped_coordinates = []
            saved_coordinates = []

            # 遍历标注数据中的区域
            for x in range(0, slide.level_dimensions[resolution_level][0], patch_size[0]):
                for y in range(0, slide.level_dimensions[resolution_level][1], patch_size[1]):
                    patch_box = (x, y, x + patch_size[0], y + patch_size[1])

                    # 计算中心坐标
                    center_x = (x + x + patch_size[0]) / 2
                    center_y = (y + y + patch_size[1]) / 2

                    # 判断是否与先前提取的视野框重叠，要求横向或纵向上的偏移正好等于20像素
                    overlap = False
                    for existing_center in extracted_coordinates:
                        existing_x, existing_y = existing_center
                        overlap_threshold = 20  # 允许的偏移阈值

                        # 计算横向和纵向上的偏移
                        x_offset = abs(existing_x - center_x)
                        y_offset = abs(existing_y - center_y)

                        if x_offset <= overlap_threshold and y_offset <= overlap_threshold:
                            overlap = True
                            break

                    if not overlap:
                        # 在这里检查在视野框能看到的最小范围的尺度的时候，是否在标注区域内
                        # 当不是最小的尺度时，无需判断标注区域，而是根据包含的标注区域的类别标签的多少进行简单的判断
                        is_inside = False
                        if resolution_level == 0 or resolution_level == 1:
                            if is_inside_annotation_center(patch_box, annotation_data):
                                is_inside = True
                                label = class_labels.get(classification_name, 3)  # 获取类别标签，如果匹配失败，使用3（None）作为默认值

                            if is_inside:
                                # 检查是否满足背景限制
                                if not is_background(x, y, resolution_level, patch_size, slide):
                                    # 在这里，将patch转换为特征张量
                                    print(f"Reading patch at {x}, {y} at resolution level {resolution_level}")
                                    patch = slide.read_region((int(x * scale), int(y * scale)), resolution_level,
                                                              patch_size)
                                    patch = Image.fromarray(np.array(patch))  # 转换为PIL图像

                                    # 调用特征提取函数
                                    feature_tensor = extract_features_from_patch(patch, model)

                                    # 获取标签
                                    matching_keys = [key for key, value in class_labels.items() if value == label]

                                    if matching_keys:
                                        thisclass = matching_keys[0]
                                        print("对应的键为:", thisclass)
                                    else:
                                        print("未找到匹配的键")

                                    # 构建文件名并保存为 .pt 文件，使用中心坐标
                                    file_name = f'{thisclass}_{label}_{wsi_file}_resolution_{resolution_level}_{int(center_x)}_{int(center_y)}.pt'
                                    file_path = os.path.join(folder_name, file_name)
                                    torch.save(feature_tensor, file_path)

                                    # 记录已保存的坐标
                                    extracted_coordinates.append((center_x, center_y))
                                    saved_coordinates.append((x, y))

                                    print(
                                        f"Saved patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                                else:
                                    skipped_coordinates.append(patch_box)
                                    print(
                                        f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                            else:
                                skipped_coordinates.append(patch_box)
                                print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: None")
                        else:
                            label = class_labels.get(classification_name, 3)
                            if not is_background(x, y, resolution_level, patch_size, slide):
                                # 在这里，将patch转换为特征张量
                                print(f"Reading patch at {x}, {y} at resolution level {resolution_level}")
                                patch = slide.read_region((int(x * scale), int(y * scale)), resolution_level, patch_size)
                                patch = Image.fromarray(np.array(patch))  # 转换为PIL图像

                                # 调用特征提取函数
                                feature_tensor = extract_features_from_patch(patch, model)

                                # 获取标签
                                matching_keys = [key for key, value in class_labels.items() if value == label]

                                if matching_keys:
                                    thisclass = matching_keys[0]
                                    print("对应的键为:", thisclass)
                                else:
                                    print("未找到匹配的键")

                                # 构建文件名并保存为 .pt 文件，使用中心坐标
                                file_name = f'{thisclass}_{label}_{wsi_file}_resolution_{resolution_level}_{int(center_x)}_{int(center_y)}.pt'
                                file_path = os.path.join(folder_name, file_name)
                                torch.save(feature_tensor, file_path)

                                # 记录已保存的坐标
                                extracted_coordinates.append((center_x, center_y))
                                saved_coordinates.append((x, y))

                                print(f"Saved patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                            else:
                                skipped_coordinates.append(patch_box)
                                print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")

                    y += patch_size[1]  # 向下移动
                    x += patch_size[0]

            # 在保存特征文件后， saved_coordinates 和 skipped_coordinates 可以进一步保存到文件

# 不需要再手动关闭幻灯片对象，已由上下文管理器处理

