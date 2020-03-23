
'''
生成一个新的csv文件 在labels_val.csv随机删掉一部分
csv文件 随机删掉一些 另存为新的csv
'''
import csv
import numpy as np

images_dir = 'H:\\DataSets\\driving_datasets\\'
val_labels_filename = 'H:\\DataSets\\driving_datasets\\labels_val.csv'

eval_labels_filename = 'H:\\DataSets\\driving_datasets\\labels_eval_dfy.csv'


with open(val_labels_filename, 'r') as csvfile:
    file_reader = csv.reader(csvfile)
    log = []
    for row in file_reader:
        log.append(row)
log = np.array(log)
# 二维矩阵里面都是字符串 (132407, 6)
print(log.shape)
print(log.dtype)
print(log[:5, :])

# 共有图片数：4241张  随机删掉 4141张图片 还剩100张图片
all_images_name = list(set(log[1:, 0]))
print('随机打乱前：', len(all_images_name), all_images_name)

np.random.shuffle(all_images_name)

print('随机打乱后：', len(all_images_name), all_images_name)

images_name_delete = all_images_name[:4141]
print('要删除的图像名字：', len(images_name_delete), images_name_delete)

# 算出log中行的index
row_index_delete = []
for img_name in images_name_delete:
    row_index_delete += list(np.argwhere(log[:, 0] == img_name).reshape(-1))
    pass
print('log中要删除的index:', len(row_index_delete), row_index_delete)
new_log = np.delete(log, obj=row_index_delete, axis=0)

print(new_log.shape)
print(new_log.dtype)
print(new_log[:5, :])
all_images_name_new = list(set(new_log[1:, 0]))
print('删除后图片总数100不变：', len(all_images_name_new), all_images_name_new)


with open(eval_labels_filename, 'w', newline='') as csvfile:
    file_writer = csv.writer(csvfile)
    # 写入多行writerows 写入单行writerow
    file_writer.writerows(new_log)

