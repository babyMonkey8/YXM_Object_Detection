'''
SSD 模型.
'''

import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.regularizers import l2
import keras.backend as K

from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from tensorflow_tanxinkeji_works.Preject4_目标检测.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

def build_model(image_size,
                n_classes,
                mode='training',
                l2_regularization=0.0,                          #l2_regularization=0.0005,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,                                    #scales = [0.08, 0.16, 0.32, 0.64, 0.96]，如果设置了这个值, 那么 `min_scale` 和 `max_scale` 会被忽略
                aspect_ratios_global=[0.5, 1.0, 2.0],           #aspect_ratios_global=[0.5, 1.0, 2.0]
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,                         #two_boxes_for_ar1 == True
                steps=None,                                     #steps = None
                offsets=None,                                   #offsets = None
                clip_boxes=False,                               #clip_boxes = False
                variances=[1.0, 1.0, 1.0, 1.0],                 #设置为1
                coords='centroids',
                normalize_coords=False,                         #normalize_coords = True 是否使用相对于图像尺寸的相对坐标
                subtract_mean=None,                             #intensity_mean = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
                divide_by_stddev=None,                          #intensity_range = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
                swap_channels=False,                            #swap_channels=False：是否想把 图像的通道转一下，在本项目没有这个必要
                confidence_thresh=0.01,                         #设置一个阈值，如果我们的神经网络输出的anchor box，这个anchor box对应的概率小于这个阈值，这个anchor box会被抛弃
                iou_threshold=0.45,                             #判断两个框的重合度，范围是[0, 1]，设置一个iou阈值，在多个anchor box都检测到同一目标物体时，去除掉与目标物体重合度低于阈值的anchor box
                top_k=200,                                      #k= 200表示保留概率最高的200个anchor bo
                nms_max_output_size=400,
                return_predictor_sizes=False):                  #除了返回模型之外，是否还返回每一个预测层输出的anchor box的大小，可以方便调试。
    '''
    此模型包含 7 个卷积层, 其中 4 个预测层, 预测层从第 4, 5, 6, 和 7 层做预测.

    '''

    n_predictor_layers = 4 # 本网络中预测层的层数。
    n_classes += 1 # 总的类别数（包括背景）。
    l2_reg = l2_regularization # 正则化参数。
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2] # 图像的高度、宽度、通道数。

    ############################################################################
    # 排除一些例外情况。
    ############################################################################
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:       #aspect_ratios_per_layer是一个list of list 结构
        if len(aspect_ratios_per_layer) != n_predictor_layers:   #如果是aspect_ratios_per_layer, 判断最外层元素是否等于4，也就是判断是否等于预测层的层数
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # 如果没有传递显式的比例因子列表，则从' min_scale '和' max_scale '计算比例因子列表。
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # 计算anchor box的相关参数。
    ############################################################################
    #  设置每个预测层中anchor box的长宽比。
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers    #将aspect_ratios结构变成list of list结构

    # 计算每个预测层中每个像素点需要预测的anchor bxo的数量。我们需要这个，这样我们就知道预测层需要多少通道。
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1为第二个框为高宽比1
            else:
                n_boxes.append(len(ar))
    else: # 如果只传递一个全局长宽比列表，那么每个预测器层的框数是相同的
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers     #将n_boxes转化为[4, 4, 4, 4]结构

    if steps is None:
        steps = [None] * n_predictor_layers    #将steps转化为[None, None, None, None]结构
    if offsets is None:
        offsets = [None] * n_predictor_layers   #将offsets转化为[None, None, None, None]结构

    ############################################################################
    # 为下面的Lambda层定义函数。
    ############################################################################
    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]], tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # 构建网络。
    ############################################################################
    x = Input(shape=(img_height, img_width, img_channels))

    # 下面的lambda层是唯一需要的，而后续的lambda层是可选的。
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)  #输出shape:由output_shape参数指定的输出shape，当使用tensorflow时可自动推断, 详细见Keras补充总结文件中问题2
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:   #  本项目中不会用到
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='cov1')(x1)#kernel_initializer:权值初始化器，he_normal：正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) #详细见SSD.总结问题5 Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence anis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)

    # 下一部分是在上面定义的基础网络上添加卷积预测层。请注意，我使用“基础网络”这个术语的方式与本文不同。
    # 在本例中，我们将有四个预测层，但是当然，您可以轻松地将其改写为任意深度的基础网络，并在基础网络上添加任意数量的预测层，只需遵循这里显示的模式即可。

    # 在卷积层4、5、6和7上构建预测层。
    # 我们在每个层上构建两个预测层:一个用于类预测(分类)，一个用于盒坐标预测(定位)
    # 我们为每个anchor box预测4个anchor box坐标，因此anchor box预测器的深度为“n_boxes * 4”

    # 'classes'的输出形状:'(batch, height, width, n_boxes * n_classes)'
    classes4 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
    classes5 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
    classes6 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(conv6)
    classes7 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes7')(conv7)

    # 'boxes'的输出形状:'(batch, height, width, n_boxes * 4)'
    boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
    boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
    boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)
    boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(conv7)

    #生成anchor box
    #'anchor '的输出形状:'(batch, height, width, n_boxes, 8)'
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)

    #Reshape的作用：整合形状，方便后面的拼接
    # Reshape the class predictions, yielding 3D tensors of shape '(batch, height * width * n_boxes, n_classes)'
    # We want the classes issolated in the last axis to perform softmax on them
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)

    # Reshape the box coordinate predictions, yielding 3D tensors of shape '(batch, height * width * n_boxes, 4)'
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape '(batch, height * width * n_boxes, 8)'
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2(n_classes or 4, respectively) are identical for all layer predictions
    # so we want to concatenate along axis 1
    # Output shape of 'classes_concat':(batch, n_boxes_total, n_classes)    #按照行合并
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped, classes5_reshaped, classes6_reshaped, classes7_reshaped])
    # Output shape of 'boxes_concat':(batch, n_boxes_total, 4)              #按照行合并
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped, boxes5_reshaped, boxes6_reshaped, boxes7_reshaped])
    # Output shape of 'anchors_concat':(batch, n_boxes_total, 8)            #按照行合并
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped, anchors5_reshaped, anchors6_reshaped, anchors7_reshaped])

    #The box coordinate predictions will go into the loss function just the way they are,
    #but for the class predictions, we'll apply a softmax activateion layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of 'predictions':(batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])




    # 添加网络结构

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes4._keras_shape[1:3],
                                    classes5._keras_shape[1:3],
                                    classes6._keras_shape[1:3],
                                    classes7._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
