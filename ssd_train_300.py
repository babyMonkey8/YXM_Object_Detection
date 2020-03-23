#
# 训练一个 SSD 网络用于识别车载摄像头捕捉的图像中的目标
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from math import ceil
from matplotlib import pyplot as plt
from tensorflow_tanxinkeji_works.Preject4_目标检测.models.keras_ssd300 import ssd_300
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_loss_function.keras_ssd_loss import SSDLoss
from tensorflow_tanxinkeji_works.Preject4_目标检测.ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_data_generator import DataGenerator
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from keras.utils import plot_model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
# 1. 设置模型参数
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

img_height = 300 # 图像的高度
img_width = 480 # 图像的宽度
img_channels = 3 # 图像的通道数
intensity_mean = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
intensity_range = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
n_classes = 5 # 正样本的类别:汽车、卡车、行人、自行车、交通灯(不包括背景)
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # Anchor 的 scaling factors. 如果设置了这个值, 那么 `min_scale` 和 `max_scale` 会被忽略；一共有四层的预测层，便会有四个anchor boxd=的大小，，因为1:1大小的anchor box有两个，便增加一个anchor box大小
aspect_ratios =  [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
# aspect_ratios = [1.0/3.0, 0.5, 1.0, 2.0, 3] # 每一个 Anchor 的长宽比
two_boxes_for_ar1 = True # 是否产生两个为长宽比为 1 的 Anchor
steps = None
# steps = [8, 16, 32, 64, 100, 300] # 可以手动设置 Anchor 的步长, 不建议使用,相对于300*480原始图片
offsets = None
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # 可以手动设置左上角 Anchor 的偏置, 不建议使用
clip_boxes = False # 是否将 Anchor 剪切到图像边界范围内
variances = [1.0, 1.0, 1.0, 1.0] # 可以将目标的坐标 scale 的参数, 建议保留 1.0
normalize_coords = True # 是否使用相对于图像尺寸的相对坐标
# swap_channels = [2, 1, 0]

# 2. 创建模型
# 如下代码完成 3 个任务:
# 2.1 调用函数 `build_model()` 创建模型.
# 2.2 可选择的加载模型权值.
# 2.3 编译模型, 设置优化器 (Adam) 和损失函数 (SSDLoss)
#
# 2.1: 加载模型

K.clear_session() # 从内存中清理曾经加载的模型.

model = ssd_300(image_size=(img_height, img_width, img_channels),    #(高、宽、通道数)
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=None,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2.2: 可选: 加载曾经的模型
weights_path = 'save_TrainmMode/ssd300_2/ssd7_epoch-09_loss-2.9362_val_loss-2.9136.h5'
# weights_path = './save_TrainmMode/weights_SSD300.hdf5'
model.load_weights(weights_path, by_name=True)

# 2.3: 设置优化器和损失函数
# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)     #像Adam里面的超参数一般是使用默认值
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=0.5)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 2.4：打印模型参数及模型可视化

model.summary()  # 输出模型各层的参数状况

# plot_model(model, 'dfy_ssd7.png', show_shapes=True)  #模型可视化


# 3. 设置训练需要的 data generators
# 3.1: 初始化两个 `DataGenerator` 对象: 一个用于训练, 一个用于validation

# 可选: 如果你的电脑内存够大, 可以将所有的图像加载到内存中, 设置 `load_images_into_memory` = `True`，这样训练会非常快。

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 3.2: 加载数据
# TODO: 设置数据集地址.

# Images
images_dir = 'driving_datasets\driving_datasets'

# Ground truth
train_labels_filename = 'driving_datasets\driving_datasets\labels_train.csv'
val_labels_filename = 'driving_datasets\driving_datasets\labels_val.csv'

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # CSV 文件前 6 列的值
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# 得到训练和validation数据集的数据的量.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("训练集的图像数量:\t{:>6}".format(train_dataset_size))
print("validation集的图像数量\t{:>6}".format(val_dataset_size))


# 3.3: 设置 batch_size.

batch_size = 10
# 3.4: 定义图像增强
# 数据增强暂时先过掉
data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),  #图像亮度的随机调整，-48是亮度浮动的下界，48是亮度浮动的上界，0.5是概率值
                                                            random_contrast=(0.5, 1.8, 0.5),   #图像对比度的随机调整，-0.5是亮度浮动的下界，1.8是亮度浮动的上界，0.5是概率值
                                                            random_saturation=(0.5, 1.8, 0.5), #图像饱和度的调整
                                                            random_hue=(18, 0.5),              #图像色调的调整
                                                            random_flip=0.5,                   #图像水平或垂直随机翻转
                                                            random_translate=((0.03,0.5), (0.03,0.5), 0.5),  #图像的平移变化
                                                            random_scale=(0.5, 2.0, 0.5),      #图像的缩放变化
                                                            n_trials_max=3,
                                                            clip_boxes=True,                   #图像平移之后是否需要剪切
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0,0,0))


# 3.5: 创建 encoder, 用于将人工标签转换为 SSD 损失函数需要的格式

# encoder 需要知道模型预测层输出的特征图的尺寸,用于产生 Anchor.
# Output shape of 'classes':'(batch, height, width, n_boxes * n_classes)'
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]
# 为什么进行编码
ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=None,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 3.6: 创建用于 Keras的 `fit_generator()` 的 generator.  用于批量化生成数据

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# 4. 训练模型
#
# 设置 Keras callbacks. 一个用于 early stopping, 一个用于在训练看起来没有进展的情况下降低学习率,
# 一个用于保存当前最佳模型, 一个用于将训练过程的值写入 CSV 文件.
#
model_checkpoint = ModelCheckpoint(filepath='save_TrainmMode\ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='save_TrainmMode\ssd7_training_log.csv',
                       separator=',',
                       append=True)

# EarlyStopping：当监测值不再改善时，该回调函数将中止训练
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,  # 增大或减小的阈值，如发现loss相比上一个epoch训练相比，没有下降这个阀值，则是没有提升
                               patience=10,  # 如发现loss相比上一个epoch训练没有下降，则经过patience个epoch后停止训练。
                               verbose=1)


# terminate_on_nan = TerminateOnNaN()
# ReduceLROnPlateau：当评价指标不在提升时，减少学习率
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,  # 每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                         patience=8,  # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                         verbose=1,
                                         epsilon=0.001,  # 阈值，用来确定是否进入检测值的“平原区”
                                         cooldown=0,  # 学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                         min_lr=0.00001)  # 学习率的下限

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]

# TODO: 设置 epochs
initial_epoch   = 0
final_epoch     = 20
steps_per_epoch = 1000


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)


# 画 training and validation loss 的走势图:
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24})
plt.show()