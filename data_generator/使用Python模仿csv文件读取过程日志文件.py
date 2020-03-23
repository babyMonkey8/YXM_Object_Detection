'''

'''
import csv
import numpy as np
import os

images_dir = 'driving_datasets/driving_datasets'
labels_filename = '../driving_datasets/driving_datasets/labels_val.csv'
include_classes = 'all'
input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
random_sample = False
# print(input_format.index('class_id'))

filenames = [] #保存图像的路径
image_ids = [] #保存图像名（不包括后缀名）
labels = []  # 保存类别和坐标信息

dataset_size = 0 # 文件中图片的数量
dataset_indices = [] # 文件中图片对应的index值

data = []
with open(labels_filename) as csvfile:
    # print('csvfile:\n', csvfile)
    csvread = csv.reader(csvfile, delimiter=',')  #reader返回的值是csv文件中每行的列表，将每行读取的值作为列表返回
    # print('csvfile_read:\n', csvfile)
    next(csvread)  # 跳过第一行
    for row in csvread:
        # print('行行行行：\n', row)
        if include_classes == 'all' or int(row[input_format.index('class_id')].strip()) in include_classes: # 如果读取包含所有类别或者class_id是数据集中包含的类之一
            box = [] #存储box类别和坐标，示例展示：['1478899046136829030.jpg', 5, 201, 129, 206, 135]
            box.append(row[input_format.index('frame')].strip()) # Select the image name column in the input format and append its content to `box`
            for element in labels_output_format: # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                box.append(int(row[input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
            data.append(box)
# print('未排序data:\n', data[:4])
data = sorted(data)  #按照‘文件名’来进行排序，sorted默认是升序，默认按照列表中第一个元素进行排序
# print('已排序data:\n', data[:10])

# 整理实际的样品和标签清单
current_file = data[0][0]  # 保存当前文件名
# print('current_file:', current_file)
current_image_id = data[0][0].split('.')[0]  # 保留图像名中‘.’之前的部分，去掉后缀
# print('current_image_id:', current_image_id)
current_labels = []  # 保存类别和坐标信息

for i, box in enumerate(data):
    # print(i)
    # print(box)
    if box[0] == current_file:  # 如果这个框(即CSV文件的这一行)属于当前图像文件
        current_labels.append(box[1:])
        if i == len(data) - 1:  # 如果这是CSV文件的最后一行
            if random_sample:  # 以防我们使用的不是完整的数据集，而是它的随机样本
                p = np.random.uniform(0, 1)
                if p >= (1 - random_sample):
                    labels.append(np.stack(current_labels, axis=0))
                    filenames.append(os.path.join(images_dir, current_file))
                    image_ids.append(current_image_id)
            else:
                labels.append(np.stack(current_labels, axis=0))
                filenames.append(os.path.join(images_dir, current_file))
                image_ids.append(current_image_id)
    else:
        if random_sample:  # 以防我们使用的不是完整的数据集，而是它的随机样本
            p = np.random.uniform(0, 1)
            if p >= (1 - random_sample):
                labels.append(np.stack(current_labels, axis=0))
                filenames.append(os.path.join(images_dir, current_file))
                image_ids.append(current_image_id)
        else:
            labels.append(np.stack(current_labels, axis=0))
            filenames.append(os.path.join(images_dir, current_file))
            image_ids.append(current_image_id)
        current_labels = []  # 重置标签列表，因为这是一个新图片
        current_file = box[0]
        current_image_id = box[0].split('.')[0]
        current_labels.append(box[1:])
        if i == len(data) - 1:  # 如果这是CSV文件的最后一行
            if random_sample:  # 以防我们使用的不是完整的数据集，而是它的随机样本
                p = np.random.uniform(0, 1)
                if p >= (1 - random_sample):
                    labels.append(np.stack(current_labels, axis=0))
                    filenames.append(os.path.join(images_dir, current_file))
                    image_ids.append(current_image_id)
            else:
                labels.append(np.stack(current_labels, axis=0))
                filenames.append(os.path.join(images_dir, current_file))
                image_ids.append(current_image_id)

print('filenames:\n', filenames[:5])
print('image_ids:\n', image_ids[:5])
print('labels:\n', labels[:5])

labels = np.array(labels) # 本项目中的labels是不规则的列表。打印出来是一维的，但实际上可以理解为二维的，第一维度是行即每张图片一行，第二维度是列，每张图片对应的所有anchorbox的类别和坐标信息
# print(labels.shape)
print(labels)
# print(labels.ndim)

dataset_size = len(filenames)
# print('dataset_size：', dataset_size)

dataset_indices = np.arange(dataset_size, dtype=np.int32)
# print('dataset_indices:', dataset_indices)




