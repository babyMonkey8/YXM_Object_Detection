
'''
SSD模型的评估流程，最终得出一个mAP值
模型的评估 给定一个keras ssd模型 和 一个数据集
计算 mean_average_precision  mAP值 AP
评估器适用于任何SSD模型和任何与DataGenerator兼容的数据集
'''

from keras import backend as K
from keras.models import load_model
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_loss_function.keras_ssd_loss import SSDLoss
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_data_generator import DataGenerator
from tensorflow_tanxinkeji_works.Preject4_目标检测.eval_utils.average_precision_evaluator_dfy import Evaluator
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_L2Normalization import L2Normalization

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
import numpy as np
# 设置一些配置参数
img_height = 300  # 图像的高度
img_width = 480  # 图像的宽度
img_channels = 3  # 图像的通道数
n_classes = 5  # 正样本的类别 (不包括背景)

# 1. 加载一个训练过的SSD模型
# 第二种情况： 1.2. Load a trained model
# model_path = './01_train_complete/dfy_ssd7_epoch-29_loss-1.8935_val_loss-1.1905.h5'
# model_path = './02_train_complete/dfy02_ssd7_epoch-32_loss-1.9307_val_loss-1.3584.h5'
# model_path = './03_train_complete/dfy03_ssd7_epoch-55_loss-2.0393_val_loss-1.5247.h5'
# 七层SSD最原始
# model_path = 'save_TrainmMode/ssd7_1/ssd7_epoch-25_loss-2.0912_val_loss-1.4135.h5'
# model_path = 'save_TrainmMode/ssd7_2/ssd7_epoch-24_loss-2.3117_val_loss-1.6521.h5'

# model_path = 'save_TrainmMode/ssd7_4/ssd7_epoch-10_loss-1.9438_val_loss-1.4201.h5'
model_path = 'save_TrainmMode/ssd7_epoch-29_loss-1.9300_val_loss-1.3761.h5'


# alpha 越大 更多的考虑 正样本的定位损失
ssd_loss = SSDLoss(neg_pos_ratio=3,
                   n_neg_min=1,
                   alpha=5.0)
K.clear_session()   # 从内存中清理曾经加载的模型.
# 这里假设你想加载的模型是使用 'training' mode 训练的. 如果你想加载的模型是使用 'inference' 或者
# 'inference_fast' mode 训练的, 你需要在`custom_objects`添加 `DecodeDetections` 或者
# `DecodeDetectionsFast` 层.


model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               # 'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

model.summary()

# 2. 为评估数据集创建一个数据生成器
val_dataset = DataGenerator(load_images_into_memory=False,
                            labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'))

# Images
images_dir = 'driving_datasets\driving_datasets'

# Ground truth

val_labels_filename = 'driving_datasets\driving_datasets\labels_eval_dfy.csv'  # 这个数据集中只有100张图片

# images_dir = 'H:\\DataSets\\driving_datasets\\'
# val_labels_filename = 'H:\\DataSets\\driving_datasets\\labels_val.csv'

# 验证数据集共4241张图片

# 评估模型的数据集共100张图片
# eval_labels_filename = 'H:\\DataSets\\driving_datasets\\labels_eval_dfy.csv'
val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

print("评价数据集的图像数量\t{:>6}".format(val_dataset.get_dataset_size()))

# 评估时候的 数据生成器
# data_generator.generate(batch_size=batch_size,
#                          shuffle=False,
#                          transformations=transformations,
#                          label_encoder=None,
#                          returns={'processed_images',
#                                   'image_ids',
#                                   'evaluation-neutral',
#                                   'inverse_transform',
#                                   'original_labels'},
#                          keep_images_without_gt=True,
#                          degenerate_box_handling='remove')
# Generate batch.生成一个批次的数据
# batch_X, batch_image_ids, batch_eval_neutral, batch_inverse_transforms, batch_orig_labels = next(generator)

# 生成日志
dfy_ssd_evaluation_result_log = open('eval_utils\dfy_ssd_evaluation_result_log.txt', 'w')

# 3. Run the evaluation
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=val_dataset,
                      model_mode='training',
                      dfy_ssd_evaluation_result_log=None)
# model_mode='training' 预测后的结果 需要解码的  我调整了解码的三个阈值(主要是置信度阈值)之后 估计mAP值会发生变化哦？
# 置信度阈值太高不行 导致预测的结果不能涵盖所有的正样本类别 最终的结果：mean_average_precision(具体的值-不算类别0): 0.2662049009269257
batch_size = 5
# batch_size = 256
results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=batch_size,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,  # 预测box匹配gt box
                    border_pixels='half',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=False,
                    return_recalls=False,
                    return_average_precisions=True,
                    verbose=True,
                    decoding_confidence_thresh=0.1,  # 解码的置信度阈值
                    decoding_iou_threshold=0.45,  # NMS 里
                    decoding_top_k=200)

mean_average_precision, average_precisions = results

print('\n最终的结果：mean_average_precision(每个类别的平均精准率再求平均值-不算类别0背景):', mean_average_precision)
class_name = ['汽车', '卡车', '行人', '自行车', '交通灯']
print('一共{}个类每个类,每个类别的平均精准率(average_precisions):'.format(len(average_precisions)))
for i in range(len(average_precisions) - 1):
    print('{}的平均精准率(average_precisions):{}'.format(class_name[i], average_precisions[i + 1]))


# 存储到日志文件中
print('\n最终的结果：mean_average_precision(每个类别的平均精准率再求平均值-不算类别0背景):', mean_average_precision, file=dfy_ssd_evaluation_result_log)
print('average_precisions(每个类别的平均精准率):', len(average_precisions), average_precisions, file=dfy_ssd_evaluation_result_log)

dfy_ssd_evaluation_result_log.close()


