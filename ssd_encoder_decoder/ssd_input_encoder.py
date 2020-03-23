'''
An encoder that converts ground truth annotations to SSD-compatible training targets.
'''

import numpy as np

from tensorflow_tanxinkeji_works.Preject4_目标检测.bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from tensorflow_tanxinkeji_works.Preject4_目标检测.ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class SSDInputEncoder:
    '''
    变换图像中目标检测的人工标注框(2D边界框坐标和类标签)到所需的格式训练一个SSD模型。

    在对人工标注框进行编码的过程中，建立了锚盒模板，并通过相交-过并阈值准则将锚盒与地面真值匹配。
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 background_id=0):
        '''
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            predictor_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional predictor layers.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be >0.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Note that the second to last of the linearly interpolated
                scaling factors will actually be the scaling factor for the last predictor layer, while the last
                scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
                if `two_boxes_for_ar1` is `True`. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be greater than or equal to `min_scale`.
            scales (list, optional): A list of floats >0 containing scaling factors per convolutional predictor layer.
                This list must be one element longer than the number of predictor layers. The first `k` elements are the
                scaling factors for the `k` predictor layers, while the last element is used for the second box
                for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
                last scaling factor must be passed either way, even if it is not being used. If a list is passed,
                this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
                Note that you should set the scaling factors such that the resulting anchor box sizes correspond to
                the sizes of the objects you are trying to detect.
            aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers. Note that you should set the aspect ratios such
                that the resulting anchor box shapes roughly correspond to the shapes of the objects you are trying to detect.
            aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
                If a list is passed, it overrides `aspect_ratios_global`. Note that you should set the aspect ratios such
                that the resulting anchor box shapes very roughly correspond to the shapes of the objects you are trying to detect.
            two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
                If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
                either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
                pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
                the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
                If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
                If no steps are provided, then they will be computed such that the anchor box center points will form an
                equidistant grid within the image dimensions.
            offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
                either floats or tuples of two floats. These numbers represent for each predictor layer how many
                pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
                as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
                of the step size specified in the `steps` argument. If the list contains floats, then that value will
                be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
                `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
            clip_boxes (bool, optional): If `True`, limits the anchor box coordinates to stay within image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            matching_type (str, optional): Can be either 'multi' or 'bipartite'. In 'bipartite' mode, each ground truth box will
                be matched only to the one anchor box with the highest IoU overlap. In 'multi' mode, in addition to the aforementioned
                bipartite matching, all anchor boxes with an IoU overlap greater than or equal to the `pos_iou_threshold` will be
                matched to a given ground truth box.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box.
            neg_iou_limit (float, optional): The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
                and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
                This means instead of using absolute tartget coordinates, the encoder will scale all coordinates to be within [0,1].
                This way learning becomes independent of the input image size.
            background_id (int, optional): Determines which class ID is for the background class.
        '''
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # 处理异常。
        ##################################################################################

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != predictor_sizes.shape[0] + 1): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes)+1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else: # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

        if not (aspect_ratios_per_layer is None):
            if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if (aspect_ratios_global is None):
                raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one offset value per predictor layer.")

        ##################################################################################
        # 设置或计算参数。
        ##################################################################################
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1 # 背景类为+ 1
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale

        # 如果“scale”为空，则通过在“min_scale”和“max_scale”之间进行线性插值来计算比例因子。然而，如果给出了一个明确的“scale”列表，那么它将超越“min_scale”和“max_scale”。
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)
        else:
            # 如果显式地给出了一个scale列表，我们将使用它而不是从' min_scale '和' max_scale '计算它。
            self.scales = scales
        # 如果“aspect_ratios_per_layer”为None，那么我们将对所有预测层使用相同的纵横比“aspect_ratios_global”列表。但是，如果给出了‘aspect_ratios_per_layer’，那么它将超越‘aspect_ratios_global’。
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            # 如果每一层都有纵横比，我们就会用到它们。
            self.aspect_ratios = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        # 计算每个预测层在每个空间位置的anchor box个数。
        # 例如，如果一个预测层有三个不同的长宽比[1.0、0.5、2.0]，并且应该预测两个长宽比1.0大小略有不同的盒子，那么该预测层在特征图的每个空间位置总共预测四个盒子。
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

        ##################################################################################
        # 计算每个预测层的anchor box。
        ##################################################################################

        # 计算每个预测层的anchor box。我们只需要这样做一次，因为anchor box只依赖于模型配置，而不是输入数据。
        # 对于每个预测器层(即对于每个比例因子)，该层anchor box的张量将具有形状' (feature_map_height, feature_map_width, n_boxes, 4) '。

        # 下面的列表只存储诊断信息。有时，在列表中加入方框的中心点、高度、宽度等是很方便的。
        self.boxes_list = [] # 存储每个预测层的anchor box的信息
        self.wh_list_diag = [] # 存储每个预测层的anchor box的宽和高度
        self.steps_diag = [] # 存储每个预测层水平和垂直方向上的步长
        self.offsets_diag = [] # 存储每个预测层水平和垂直方向上的offset
        self.centers_diag = [] # 存储每个预测层的anchorbox的中心点的cx,cy坐标信息

        # 遍历所有预测层并计算每个预测层的anchorbox边界框。
        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   next_scale=self.scales[i+1],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i],
                                                                                   diagnostics=True)  # diagnostics决定是否返回所有参数
            #返回参数解释：
            # boxes = boxes_tensor(height,width,n_boxes(4),4)
            # center = (cy, cx)元组
            # wh = wh_list 列表：[anchorbox的高度，banchorbox的宽度]
            # step = (step_height, step_width)
            # offset = (offset_height, offset_width)

            self.boxes_list.append(boxes)  # list :shape = (4, height, width, n_boxes(4), 4 )
            self.wh_list_diag.append(wh)  # list: shape=(4, 2)
            self.steps_diag.append(step)  # list：[(), (), (), ()]
            self.offsets_diag.append(offset)  # list: shape=(4, 2)
            self.centers_diag.append(center)  # list: [(), (), (), ()]

    def __call__(self, ground_truth_labels, diagnostics=False):
        '''
        Converts ground truth bounding box data into a suitable format to train an SSD model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)` (i.e. the 'corners' coordinate format), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.

                长度' batch_size '的python列表，其中包含每个批处理图像的一个2D Numpy数组。
                每一个这样的数组有“k”行“k”个人工标注框属于各自的图像,每一个人工标注框的格式”(class_id、xmin ymin, xmax, ymax)”(即“角落”坐标格式)
                和“class_id”必须是大于0的整数为所有盒子类ID 0是预留给后台类。
                即：csv文件读取数据并存储在labels变量 == ground_truth_labels ，区别：labels是所有图片，而ground_truth_labels是batchsizez张图片
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        '''

        # 映射来定义哪些索引表示在人工标注框中的哪些坐标
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        #批处理的大小
        batch_size = len(ground_truth_labels)

        ##################################################################################
        # 生成一个临时的模板，形状=y_predict
        ##################################################################################
        # y_encoded.shape = (batchsize, feature_map_height * feature_map_width * n_boxes, 18)
        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False) # 18中产生的前六个维度是独热编码

        ##################################################################################
        # 为图片的人工标注框匹配anchorboxes
        ##################################################################################
        # 将人工标注框与anchor box匹配。每一个anchor box没有一个人工标注框匹配，并且最大IoU与任何人工标注框重叠小于或等于' neg_iou_limit '的将是一个负(背景)框。
        y_encoded[:, :, self.background_id] = 1 # 默认情况下，所有的框都是背景框
        n_boxes = y_encoded.shape[1] # 每张图片预测的anchorboxes数量
        class_vectors = np.eye(self.n_classes) # 生成一个对角矩阵，对角线上是1，其余都是0，用于独热编码mask

        for i in range(batch_size): # 循环batch_size中的每张图片

            if ground_truth_labels[i].size == 0: continue # 如果该图片没有人工标注框，就不为该图片匹配
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item
            # 在尝试任何计算之前，检查退化的人工标注边界框.
            if np.any(labels[:,[xmax]] - labels[:,[xmin]] <= 0) or np.any(labels[:,[ymax]] - labels[:,[ymin]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                         "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. Degenerate ground truth " +
                                         "bounding boxes will lead to NaN errors during the training.")

            #或许可以将方框坐标标准化。
            if self.normalize_coords:
                labels[:,[ymin,ymax]] /= self.img_height # 将ymin和ymax相对于图像高度进行归一化
                labels[:,[xmin,xmax]] /= self.img_width # 将xmin和xmax相对于图像宽度进行标准化

            # Maybe convert the box coordinate format.
            if self.coords == 'centroids':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2centroids', border_pixels=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2minmax')

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]  # 将一张图片中的人工标注框的类别转独热编码
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin,ymin,xmax,ymax]]], axis=-1)

            # 计算所有anchorboxes和人工标注框之间的IoU相似性。
            # 这个矩阵的形状： `(num_ground_truth_boxes, num_anchor_boxes)`.  #这里的iou支持多种框
            similarities = iou(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,-12:-8], coords=self.coords, mode='outer_product', border_pixels=self.border_pixels)  # 返回 shape = m * n， values 是iou 面积


            # 首先:进行严格匹配，即将每个人工标注框匹配到一个具有最高IoU的anchor box。
            #        这将确保每个人工标注框将至少有一个良好的匹配。
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities) # 返回的最匹配的anchhor box 下标

            # 将人工标注框的数据写入匹配到的anchor box中
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            #将匹配的anchor box的列的iou值设置为0，表示它们已经匹配。
            similarities[:, bipartite_matches] = 0

            #第二:也许做“多重”匹配，每个剩余的锚框将匹配到它的最相似的人工标注框，至少有一个“pos_iou_threshold”的IoU，或不匹配，如果没有这样的人工标注框框。
            if self.matching_type == 'multi':

                # 获取所有满足IoU阈值的匹配。
                # match_multi返回两个长度相等的一维数字数组，表示匹配的索引。第一个数组包含' weight_matrix '的第一个轴上的索引，例如[0 0 0]；第二个数组包含沿第二轴的指标[775 779 783]
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

                # 将人工标注框数据写入匹配的anchor box中
                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

                # 将匹配的anchor box的列的iou值设置为0，表示它们已经匹配。
                similarities[:, matches[1]] = 0

            # 第三:匹配完成后，所有的负(背景)anchor box有一个IoU ' neg_iou_limit '或更多的人工标注框将被设置为中立样本，即他们将不再是背景框。
            # 这些anchore box“太接近”人工标注框，不能作为有效的背景框。即从剩余的负样本中过滤掉中立样本

            max_background_similarities = np.amax(similarities, axis=0)   #numpy.amax() 用于计算数组中的元素沿指定轴的最大值。
            # 补充：在np.argmax(similarities, axis=0)中，为什么使用axis =0方式而不是直接使用axis = 1同时>=阈值的方法
                # 因为我们的出发点是想删除中立anchor box样本（我们已经找出了正样本），保留负样本作为背景框，那么首先先为每个anchor box匹配到最佳的人工标注框，然后再根据阈值过滤掉这些中立anchor box
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]  # numpy.nonzero() 函数返回输入数组中非零元素的索引。
            y_encoded[i, neutral_boxes, self.background_id] = 0

        ##################################################################################
        # 将绝对坐标转化为相对坐标
        ##################################################################################

        if self.coords == 'centroids':  #  转化为相对值
            y_encoded[:,:,[-12,-11]] -= y_encoded[:,:,[-8,-7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:,:,[-12,-11]] /= y_encoded[:,:,[-6,-5]] * y_encoded[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:,:,[-10,-9]] /= y_encoded[:,:,[-6,-5]] # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encoded[:,:,[-2,-1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        elif self.coords == 'corners':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-10]] /= np.expand_dims(y_encoded[:,:,-6] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-11,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-7], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
        elif self.coords == 'minmax':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-11]] /= np.expand_dims(y_encoded[:,:,-7] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-10,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-6], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

        if diagnostics:   # 使得负样本和中立样本全为0，保留正样本，只有正样本的转换才有意义
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,-12:-8] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        '''
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics (bool, optional): If true, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`
                4) A tuple containing `(offset_height, offset_width)`
                This information can be useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor


    def generate_encoding_template(self, batch_size, diagnostics=False):
        '''
        为给定批的人工标注框张量生成一个编码模板。

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the SSD model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes and the 4 variance values.
        '''
        # 在所有批处理项上为每个预测器层平铺anchor box。
        boxes_batch = []
        for boxes in self.boxes_list:
            # 为“自我”添加一个维度。boxes_list ':解释批处理大小并将其平铺。
            # 结果是一个5维的形状张量(batch_size, feature_map_height, feature_map_width, n_boxes, 4)
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
            # order of the tensor content will be identical to the order obtained from the reshaping operation
            # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)  # boxes_batch.shape = (batchsize, feature_map_height * feature_map_width * n_boxes, 4)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)  # boxes_batch.shape = (batchsize, feature_map_height * feature_map_width * n_boxes, 4)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encoding_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
