'''
    测试模型
'''
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_loss_function.keras_ssd_loss import SSDLoss
from keras import backend as K
from keras.models import load_model
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_data_generator import DataGenerator
import matplotlib.pyplot as plt
from tensorflow_tanxinkeji_works.Preject4_目标检测.ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_L2Normalization import L2Normalization
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
from tensorflow_tanxinkeji_works.Preject4_目标检测.eval_utils.average_precision_evaluator import Evaluator

# ## 1. 设置模型参数
#
# 这些参数同时需要在 `build_model()` 和 `SSDInputEncoder` 的构造函数中使用
#
# * 设置图像的高度, 宽度, 和色彩通道数量
# * 设置正样本类别个数 (不包括背景).
# * `mode` 用于 `build_model()` 函数, 决定是否使用 `DecodeDetections` 层作为最后一层. Mode 的值为 'training' 时,
# * 网络输出原始预测值. 当值为 'inference' 或者 'inference_fast' 时, 原始的输出被转为了绝对坐标, 使用了概率筛选, 使用
# * 了 non-maximum suppression, 以及 top-k 筛选. 'inference' 使用了原始Caffe实现的算法, 'inference_fast' 使用了
# * 更快, 但是相对不太精确的算法
# * 本例子只有 4 个预测层, 但是需要 5 个 scaling factors. 因为最后一个 scaling factor 被用于计算最后一层的第二个长宽比为 1 的 Anchor 的尺寸.
# * `build_model()` 与 `SSDInputEncoder` 有两个 Anchor 的长宽比: `aspect_ratios_global` 和 `aspect_ratios_per_layer`.
# * 你只需要设置其中一个. 如果你想使用 `aspect_ratios_global`, 设置其值为一个Python list, 这个值会被每一层使用.
# * 如果想使用`aspect_ratios_per_layer`, 设置其值为包含 Python list 的 Python list. 每一个 list 包含每一层的 长宽比.
# * 如果 `two_boxes_for_ar1 == True`, 那么预测层为为长宽比为一的情况预测两个边界框, 一个大一些, 一个小一些
# * 如果 `clip_boxes == True`, 那么 Anchor 会被剪切到图像边界内, 建议设置为 False.
# * 在训练的时候, 边界框相对于 Anchor 的偏置需要除以 variances, 设置为 1.0, 相当于没有使用. 使用小于 1.0 的值, 相当于使得这个维度的分布值变大.
# * `normalize_coords` 为 True 的时候, 会将所有绝对坐标值转换为相对于图像宽度和高度的值. 这个设置不影响最终的输出.

# 1. 设置模型参数
img_height = 300 # 图像的高度
img_width = 480 # 图像的宽度
img_channels = 3 # 图像的通道数
intensity_mean = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
intensity_range = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
n_classes = 5 # 正样本的类别:汽车、卡车、行人、自行车、交通灯(不包括背景)
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # Anchor 的 scaling factors. 如果设置了这个值, 那么 `min_scale` 和 `max_scale` 会被忽略；一共有四层的预测层，便会有四个anchor boxd=的大小，，因为1:1大小的anchor box有两个，便增加一个anchor box大小
# scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
aspect_ratios = [0.5, 1.0, 2.0] # 每一个 Anchor 的长宽比
# aspect_ratios =  [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True # 是否产生两个为长宽比为 1 的 Anchor
steps = None # 可以手动设置 Anchor 的步长, 不建议使用
# steps = [8, 16, 32, 64, 100, 300]
offsets = None # 可以手动设置左上角 Anchor 的偏置, 不建议使用
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False # 是否将 Anchor 剪切到图像边界范围内
variances = [1.0, 1.0, 1.0, 1.0] # 可以将目标的坐标 scale 的参数, 建议保留 1.0

normalize_coords = True # 是否使用相对于图像尺寸的相对坐标



# 2、加载一个已经训练过的模型
#
# 这里假设你想加载的模型是使用 'training' mode 训练的. 如果你想加载的模型是使用 'inference' 或者
# 'inference_fast' mode 训练的, 你需要在`custom_objects`添加 `DecodeDetections` 或者 `DecodeDetectionsFast` 层.
#
# TODO: 设置要加载的模型的路径.
# model_path = 'save_TrainmMode/ssd7_1/ssd7_epoch-25_loss-2.0912_val_loss-1.4135.h5'

# model_path = 'save_TrainmMode/ssd7_3/ssd7_epoch-12_loss-1.9432_val_loss-1.2533.h5'

# model_path= 'save_TrainmMode/ssd7_epoch-01_loss-11.0600_val_loss-8.4513.h5'

model_path = 'save_TrainmMode/ssd300_2/ssd7_epoch-10_loss-2.9544_val_loss-2.9608.h5'
# 创建 SSDLoss 对象
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # 从内存中清理曾经加载的模型.

# 加载一个已经训练过的模型,并自定义AnchorBoxes, loss
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                              'L2Normalization': L2Normalization,
                                              'compute_loss': ssd_loss.compute_loss})


print('model:', model)
print('model的类型', type(model))



# 3. 设置训练需要的 data generators
# 3.1: 初始化一个 `DataGenerator` 对象: 用于validation测试
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 3.2: 加载数据
# TODO: 设置数据集地址.
# Images
images_dir = 'driving_datasets\driving_datasets'

# Ground truth
val_labels_filename = 'driving_datasets\driving_datasets\labels_val.csv'

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')


# 得到validation数据集的数据的量.
val_dataset_size   = val_dataset.get_dataset_size()
print("validation集的图像数量\t{:>6}".format(val_dataset_size))




# 3.3: 初始化 generator

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)

# 4. 做预测
#
# 这里我们使用已经设置好的 validation generator 做预测
#
# 4.1: 获取预测所需输入
# batch_images是图片数据
# batch_labels是对应图片的标签（未编码标签）
# batch_filenames是对应图片的路径
batch_images, batch_labels, batch_filenames = next(predict_generator)

i = 0  # 图像位置

print("图像名:", batch_filenames[i])
print()
print("人工标注的值:")
print('   类别 概率 xmin ymin xmax ymax')
print(batch_labels[i])

# 4.3: 作预测

y_pred = model.predict(batch_images)

# 4.4: 解码 `y_pred`
# 如果我们训练是设置的是 'inference' 或者 'inference_fast' mode, 那么模型的最后一层为 `DecodeDetections` 层,
# `y_pred` 就无需解码了. 但是我们选择了 'training' mode, 模型的原始输出需要解码. 这就是 `decode_detections()`
# 这个函数的功能. 这个函数的功能和 `DecodeDetections` 层做的事情一样, 只是使用 Numpy 而不是 TensorFlow 实现.
# (Nunpy 只能使用CPU, 而不是GPU).
#

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.2,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)


np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("预测值:")
print('   类别   概率   xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])


# 最后, 我们可以将预测的边界框画在图像上. 每一个预测的边界框都有类别名称和概率显示在边上.
# 人工标准的边界框也使用绿色的框画出来, 便于比较.
# 4.5: 在图像上画边界框

plt.figure(figsize=(20, 12))
plt.imshow(batch_images[i])

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()  # 设置边界框的颜色
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']  # 类别的名称

# 画人工标注的边界框
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=1))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# 画预测的边界框
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='blue', fill=False, linewidth=1))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': 'blue', 'alpha': 1.0})

plt.show()
