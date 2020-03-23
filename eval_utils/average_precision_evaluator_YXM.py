
from __future__ import division
import numpy as np
from math import ceil
from tqdm import trange
import sys
import warnings

from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_data_generator import DataGenerator
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_geometric_ops import Resize
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from tensorflow_tanxinkeji_works.Preject4_目标检测.ssd_encoder_decoder.ssd_output_decoder import decode_detections
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from tensorflow_tanxinkeji_works.Preject4_目标检测.bounding_box_utils.bounding_box_utils import iou


class Evaluator:
    '''
    计算给定数据集上给定Keras SSD模型的平均精度。

    Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
    and post-2010 (integration) algorithm versions.

    Optionally also returns the average precisions, precisions, and recalls.

    The algorithm is identical to the official Pascal VOC pre-2010 detection evaluation algorithm
    in its default settings, but can be cusomized in a number of ways.
    '''

    def __init__(self,
                 model,
                 n_classes,
                 data_generator,
                 model_mode='training',
                 dfy_ssd_evaluation_result_log=None,
                 pred_format={'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5},
                 gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            model (Keras model): A Keras SSD model object.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
            model_mode (str, optional): The mode in which the model was created, i.e. 'training', 'inference' or 'inference_fast'.
                This is needed in order to know whether the model output is already decoded or still needs to be decoded. Refer to
                the model documentation for the meaning of the individual modes.
            pred_format (dict, optional): A dictionary that defines which index in the last axis of the model's decoded predictions
                contains which bounding box coordinate. The dictionary must map the keywords 'class_id', 'conf' (for the confidence),
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis.
            gt_format (list, optional): A dictionary that defines which index of a ground truth bounding box contains which of the five
                items class ID, xmin, ymin, xmax, ymax. The expected strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
        '''

        if not isinstance(data_generator, DataGenerator):
            warnings.warn("`data_generator` is not a `DataGenerator` object, which will cause undefined behavior.")

        self.model = model
        self.data_generator = data_generator
        self.n_classes = n_classes
        self.model_mode = model_mode
        self.pred_format = pred_format
        self.gt_format = gt_format
        self.dfy_ssd_evaluation_result_log = dfy_ssd_evaluation_result_log

        # 下面的列表都包含每个类的数据，即所有列表的长度' n_classes + 1 '，其中一个元素是背景类，即该元素只是一个虚拟的条目。

        # self.prediction_results：将所有图片的预测值经过解码再经过数据修整后按照类别存储在self.prediction_results，shape = (6, ),6代表6个类别
        # 每个类别中存储着该预测到的所有该类别的anchor box标签，anchorbox标签.shape = (image_id, confidence, xmin, ymin, xmax, ymax)
        self.prediction_results = None
        ##获取每个类的人工标注框的总数，并将每类人工标注框的个数存储在mum_gt_per_class,形成一个列表
        self.num_gt_per_class = None
        self.true_positives = None
        self.false_positives = None
        self.cumulative_true_positives = None
        self.cumulative_false_positives = None
        # cumulative 累计
        self.cumulative_precisions = None
        # “累计”表示每个列表中的第i个元素表示该类的第一个最高状态预测的精度。
        self.cumulative_recalls = None
        # “累计”表示每个列表中的第i个元素表示该类的第一个最高可靠预测的回忆。
        self.average_precisions = None
        self.mean_average_precision = None

    def __call__(self,
                 img_height,
                 img_width,
                 batch_size,
                 data_generator_mode='resize',
                 round_confidences=False,
                 matching_iou_threshold=0.5,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 average_precision_mode='sample',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,  #忽略中立的样本box
                 return_precisions=False,
                 return_recalls=False,
                 return_average_precisions=False,
                 verbose=True,
                 decoding_confidence_thresh=0.01,
                 decoding_iou_threshold=0.45,
                 decoding_top_k=200,
                 decoding_pred_coords='centroids',
                 decoding_normalize_coords=True):
        '''
        计算给定数据集上给定Keras SSD模型的平均精度。

        Optionally also returns the averages precisions, precisions, and recalls.

        All the individual steps of the overall evaluation algorithm can also be called separately
        (check out the other methods of this class), but this runs the overall algorithm all at once.

        Arguments:
            img_height (int): The input image height for the model.
            img_width (int): The input image width for the model.
            batch_size (int): The batch size for the evaluation.
            data_generator_mode (str, optional): Either of 'resize' and 'pad'. If 'resize', the input images will
                be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
                If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
                and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
            round_confidences (int, optional): `False` or an integer that is the number of decimals that the prediction
                confidences will be rounded to. If `False`, the confidences will not be rounded.
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            average_precision_mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision
                will be computed according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled
                for `num_recall_points` recall values. In the case of 'integrate', the average precision will be computed according to the
                Pascal VOC formula that was used from VOC 2010 onward, where the average precision will be computed by numerically integrating
                over the whole preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just
                the limit case of 'sample' mode as the number of sample points increases.
            num_recall_points (int, optional): The number of points to sample from the precision-recall-curve to compute the average
                precisions. In other words, this is the number of equidistant recall values for which the resulting precision will be
                computed. 11 points is the value used in the official Pascal VOC 2007 detection evaluation algorithm.
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `False`, even boxes that are annotated as neutral will be counted into the evaluation. If `True`,
                neutral boxes will be ignored for the evaluation. An example for evaluation-neutrality are the ground truth boxes
                annotated as "difficult" in the Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            return_precisions (bool, optional): If `True`, returns a nested list containing the cumulative precisions for each class.
            return_recalls (bool, optional): If `True`, returns a nested list containing the cumulative recalls for each class.
            return_average_precisions (bool, optional): If `True`, returns a list containing the average precision for each class.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            decoding_confidence_thresh (float, optional): Only relevant if the model is in 'training' mode.
                A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered
                for the non-maximum suppression stage for the respective class. A lower value will result in a larger part of the
                selection process being done by the non-maximum suppression stage, while a larger value will result in a larger
                part of the selection process happening in the confidence thresholding stage.
            decoding_iou_threshold (float, optional): Only relevant if the model is in 'training' mode. A float in [0,1].
                All boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
                from the set of predictions for a given class, where 'maximal' refers to the box score.
            decoding_top_k (int, optional): Only relevant if the model is in 'training' mode. The number of highest scoring
                predictions to be kept for each batch item after the non-maximum suppression stage.
            decoding_input_coords (str, optional): Only relevant if the model is in 'training' mode. The box coordinate format
                that the model outputs. Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            decoding_normalize_coords (bool, optional): Only relevant if the model is in 'training' mode. Set to `True` if the model
                outputs relative coordinates. Do not set this to `True` if the model already outputs absolute coordinates,
                as that would result in incorrect coordinates.

        Returns:
            A float, the mean average precision, plus any optional returns specified in the arguments.
        '''

        #############################################################################################
        # 对整个数据集进行预测。
        #############################################################################################
        # predict_on_dataset作用是将预测框附加到类别的的结果列表中。
        self.predict_on_dataset(img_height=img_height,
                                img_width=img_width,
                                batch_size=batch_size,
                                data_generator_mode=data_generator_mode,
                                decoding_confidence_thresh=decoding_confidence_thresh,
                                decoding_iou_threshold=decoding_iou_threshold,
                                decoding_top_k=decoding_top_k,
                                decoding_pred_coords=decoding_pred_coords,
                                decoding_normalize_coords=decoding_normalize_coords,
                                decoding_border_pixels=border_pixels,
                                round_confidences=round_confidences,
                                verbose=verbose,
                                ret=False)

        #############################################################################################
        # 获取每个类的人工标注框的总数。
        #############################################################################################
        # get_num_gt_per_class作用：获取每个类的人工标注框的总数。
        num_gt_per_class = self.get_num_gt_per_class(ignore_neutral_boxes=ignore_neutral_boxes,
                                  verbose=False,
                                  ret=True)
        if self.dfy_ssd_evaluation_result_log:
            # [  0 466  18  95   5  86] 跟csv文件中统计的一样
            print('在所有评估数据集中每个类别的gt box人工标注框的数量(放self.num_gt_per_class)跟csv文件中统计的一样：', num_gt_per_class,
                  file=self.dfy_ssd_evaluation_result_log)

        #############################################################################################
        # Match predictions to ground truth boxes for all classes.
        # 判断每个类别预测的所有box 是否正确 错误？ 首先看类别 其次看iou
        # match_predictions得到以下四个值true_pos、false_pos、cumulative_true_pos、cumulative_false_pos，例子如下所示：
        # true_pos= [1 0 0 1 1 1 0 0 0 0 1 1 0 0 1 0 1 0]
        # false_pos= [0 1 1 0 0 0 1 1 1 1 0 0 1 1 0 1 0 1]
        # cumulative_true_pos = [1 1 1 2 3 4 4 4 4 4 5 6 6 6 7 7 8 8]
        # cumulative_false_pos= [ 0  1  2  2  2  2  3  4  5  6  6  6  7  8  8  9  9 10]
        #############################################################################################
        self.match_predictions(ignore_neutral_boxes=ignore_neutral_boxes,
                               matching_iou_threshold=matching_iou_threshold,
                               border_pixels=border_pixels,
                               sorting_algorithm=sorting_algorithm,
                               verbose=verbose,
                               ret=False)

        #############################################################################################
        # 计算所有类的累积精度和召回率。
        #############################################################################################

        self.compute_precision_recall(verbose=verbose, ret=False)

        #############################################################################################
        # 计算该类的平均精度。
        #############################################################################################

        self.compute_average_precisions(mode=average_precision_mode,
                                        num_recall_points=num_recall_points,
                                        verbose=verbose,
                                        ret=False)

        #############################################################################################
        # 计算平均精度
        #############################################################################################

        mean_average_precision = self.compute_mean_average_precision(ret=True)

        #############################################################################################

        # Compile the returns.
        if return_precisions or return_recalls or return_average_precisions:
            ret = [mean_average_precision]
            if return_average_precisions:
                ret.append(self.average_precisions)
            if return_precisions:
                ret.append(self.cumulative_precisions)
            if return_recalls:
                ret.append(self.cumulative_recalls)
            return ret
        else:
            return mean_average_precision

    def predict_on_dataset(self,
                           img_height, # 300
                           img_width, # 480
                           batch_size, # 5
                           data_generator_mode='resize',
                           decoding_confidence_thresh=0.01,
                           decoding_iou_threshold=0.45,
                           decoding_top_k=200,
                           decoding_pred_coords='centroids',
                           decoding_normalize_coords=True,
                           decoding_border_pixels='include',
                           round_confidences=False,
                           verbose=True,
                           ret=False):
        '''
        在‘data_generator’给出的整个数据集上运行给定模型的预测。

        Arguments:
            img_height (int): The input image height for the model.
            img_width (int): The input image width for the model.
            batch_size (int): The batch size for the evaluation.
            data_generator_mode (str, optional): Either of 'resize' and 'pad'. If 'resize', the input images will
                be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
                If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
                and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
            decoding_confidence_thresh (float, optional): Only relevant if the model is in 'training' mode.
                A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered
                for the non-maximum suppression stage for the respective class. A lower value will result in a larger part of the
                selection process being done by the non-maximum suppression stage, while a larger value will result in a larger
                part of the selection process happening in the confidence thresholding stage.
            decoding_iou_threshold (float, optional): Only relevant if the model is in 'training' mode. A float in [0,1].
                All boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
                from the set of predictions for a given class, where 'maximal' refers to the box score.
            decoding_top_k (int, optional): Only relevant if the model is in 'training' mode. The number of highest scoring
                predictions to be kept for each batch item after the non-maximum suppression stage.
            decoding_input_coords (str, optional): Only relevant if the model is in 'training' mode. The box coordinate format
                that the model outputs. Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            decoding_normalize_coords (bool, optional): Only relevant if the model is in 'training' mode. Set to `True` if the model
                outputs relative coordinates. Do not set this to `True` if the model already outputs absolute coordinates,
                as that would result in incorrect coordinates.
            round_confidences (int, optional): `False` or an integer that is the number of decimals that the prediction
                confidences will be rounded to. If `False`, the confidences will not be rounded.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the predictions.

        Returns:
            None by default. Optionally, a nested list containing the predictions for each class.
        '''

        class_id_pred = self.pred_format['class_id'] # 0
        conf_pred = self.pred_format['conf'] # 1
        xmin_pred = self.pred_format['xmin'] # 2
        ymin_pred = self.pred_format['ymin'] # 3
        xmax_pred = self.pred_format['xmax'] # 4
        ymax_pred = self.pred_format['ymax'] # 5

        #############################################################################################
        # 为评估配置数据生成器。
        #############################################################################################
        #  将1通道和4通道图像转换为3通道图像。对已经有3个频道的图像不做任何操作。对于4通道的图像，第4通道将被丢弃。
        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height, width=img_width, labels_format=self.gt_format)
        if data_generator_mode == 'resize':
            # 图像数据的一些转换
            transformations = [convert_to_3_channels,
                               resize]
        elif data_generator_mode == 'pad':
            random_pad = RandomPadFixedAR(patch_aspect_ratio=img_width / img_height, labels_format=self.gt_format)
            transformations = [convert_to_3_channels,
                               random_pad,
                               resize]
        else:
            raise ValueError("`data_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(
                data_generator_mode))

        # 设置生成器参数（返回数据是一个批次一个批次的返回，生成器调用一次，返回一次）。
        # 图像有一些变换的策略 但是人工标注的标签是不需要编码的
        generator = self.data_generator.generate(batch_size=batch_size,
                                                 shuffle=False,
                                                 transformations=transformations,
                                                 label_encoder=None,
                                                 returns={'processed_images', # shape = (5, 300, 480, 3) 存储图片数据
                                                          'image_ids', # batch_image_ids.shape = (5, ) # 存储批处理的图片的图片名（不包括后缀名）
                                                          'evaluation-neutral', # None
                                                          'inverse_transform', # 可以暂时理解成数据增强
                                                          'original_labels'}, # batch_original_labels.shape = (5, ),相当于存储批处理的图片的类别和坐标信息， 坐标格式是（xmin, ymin, xmax, ymax）
                                                 keep_images_without_gt=True,
                                                 degenerate_box_handling='remove')

        # 如果没有任何实际的图像id，则生成伪图像id。
        # 这只是为了让求值器与有或没有图像id的数据集兼容。
        if self.data_generator.image_ids is None:
            self.data_generator.image_ids = list(range(self.data_generator.get_dataset_size()))

        #############################################################################################
        # 对数据集的所有批次进行预测并存储预测。
        #############################################################################################
        # 我们必须为每个类生成一个单独的结果列表。
        results = [list() for _ in range(self.n_classes + 1)] # results = [[], [], [], [], [], []]

        # 创建一个将图像id映射到地面真相注释的字典。我们需要它在下面。
        image_ids_to_labels = {}

        # 计算遍历整个数据集的批数。
        n_images = self.data_generator.get_dataset_size() # 整个数据集图片的个数
        n_batches = int(ceil(n_images / batch_size)) # 批次
        if verbose:
            print("图像数量 in the evaluation dataset: {}".format(n_images))
            print()
            tr = trange(n_batches, file=sys.stdout)  # tqdm的快捷方式
            tr.set_description('Producing predictions batch-wise批量生成预测')
        else:
            tr = range(n_batches)

        # Loop over all batches.
        for ii, jj in enumerate(tr):
            # Generate batch. 100张图片 batch_size=5 20个loop
            batch_X, batch_image_ids, batch_eval_neutral, batch_inverse_transforms, batch_orig_labels = next(generator)
            # Predict. 模型预测就这一行
            y_pred = self.model.predict(batch_X) # 批处理，一次预测了5张图片
            if self.dfy_ssd_evaluation_result_log and (ii == 3):
                print('开始准备评估数据：第%d批次生成器生成共%d张图片的数据：' % (ii, batch_size), file=self.dfy_ssd_evaluation_result_log)
                print('batch_X.shape(0-255之间原始图像):', batch_X.shape, file=self.dfy_ssd_evaluation_result_log)
                print('batch_image_ids:', batch_image_ids, file=self.dfy_ssd_evaluation_result_log)
                print('batch_eval_neutral=None:', batch_eval_neutral, file=self.dfy_ssd_evaluation_result_log)
                print('batch_inverse_transforms:', batch_inverse_transforms, file=self.dfy_ssd_evaluation_result_log)
                print('batch_orig_labels(人工标注框):', batch_orig_labels, file=self.dfy_ssd_evaluation_result_log)
                print('y_pred(最原始的预测输出矩阵，需要解码):', y_pred.shape, file=self.dfy_ssd_evaluation_result_log)

            # 如果模型是在“训练”模式下创建的，则需要对原始预测进行解码和过滤，否则就已经处理好了。
            if self.model_mode == 'training':
                # Decode.一次性解码一个批次的图片 shape = (5, ),每张图片有一个或者多个anchor box,每个anchor box.shape = (6,), 即（'class_id', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'）
                y_pred = decode_detections(y_pred,
                                           confidence_thresh=decoding_confidence_thresh,
                                           iou_threshold=decoding_iou_threshold,
                                           top_k=decoding_top_k,
                                           input_coords=decoding_pred_coords,
                                           normalize_coords=decoding_normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           border_pixels=decoding_border_pixels)
                if self.dfy_ssd_evaluation_result_log and (ii == 3):
                    print('第%d批次预测矩阵解码后(看10个)：' % ii, len(y_pred), np.vstack(y_pred).shape,
                          np.vstack(y_pred)[:10, :], file=self.dfy_ssd_evaluation_result_log)
            else:
                # Filter out the all-zeros dummy elements of `y_pred`.
                y_pred_filtered = []
                for i in range(len(y_pred)):
                    y_pred_filtered.append(y_pred[i][y_pred[i, :, 0] != 0])
                y_pred = y_pred_filtered
            # 转换原图像的预测框坐标。
            # 解码之后 做这个是干什么？ 关键在 batch_inverse_transforms 逆变换
            y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)  # 对预测出的坐标值进行修整
            if self.dfy_ssd_evaluation_result_log and (ii == 3):
                print('第%d批次预测矩阵逆变换后(看10个-有点区别)：' % ii, len(y_pred), np.vstack(y_pred).shape,
                      np.vstack(y_pred)[:10, :], file=self.dfy_ssd_evaluation_result_log)

            # 遍历所有批处理项。
            for k, batch_item in enumerate(y_pred):
                # 一张一张图片来
                image_id = batch_image_ids[k] # 获得一张图片的图片名（没有后缀名）
                if self.dfy_ssd_evaluation_result_log and (ii == 3):
                    print('第%d批次预测的最终结果中第%d张图片id为：%s' % (ii, k, image_id), file=self.dfy_ssd_evaluation_result_log)
                    print('该图片的模型预测结果(类别 conf box坐标)：', batch_item, file=self.dfy_ssd_evaluation_result_log)

                for box in batch_item: # 遍历一张图片中的每个anchor box
                    class_id = int(box[class_id_pred])
                    # 将方框坐标四舍五入以减少所需的内存。
                    if round_confidences:
                        confidence = round(box[conf_pred], ndigits=round_confidences)  # ndigits 四舍五入的保留位数
                    else:
                        # 置信度的精度 不改变
                        confidence = box[conf_pred]
                    xmin = round(box[xmin_pred], ndigits=1)
                    ymin = round(box[ymin_pred], ndigits=1)
                    xmax = round(box[xmax_pred], ndigits=1)
                    ymax = round(box[ymax_pred], ndigits=1)
                    prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
                    # 将预测框附加到其类的结果列表中。
                    results[class_id].append(prediction) # 按照类别放好

        # 将所有图片的预测值经过解码再经过数据修整后按照类别存储在self.prediction_results，shape = (6, ),6代表6个类别
        # 每个类别中存储着该预测到的所有该类别的anchor box标签，anchorbox标签.shape = (image_id, confidence, xmin, ymin, xmax, ymax)
        self.prediction_results = results
        if self.dfy_ssd_evaluation_result_log:
            print('共100张图片 一个批次5张 每张图片有若干个box（class_id,conf,坐标） 按类别id放好(image_id, conf,坐标)', file=self.dfy_ssd_evaluation_result_log)
            print('评估数据集中模型预测的所有结果放self.prediction_results(按类别数%d):' % len(results), file=self.dfy_ssd_evaluation_result_log)
            print('类别为0 背景的box个数：', len(results[0]), results[0][:10], file=self.dfy_ssd_evaluation_result_log)
            print('类别为1 预测为：汽车的box个数：', len(results[1]), results[1][:10], file=self.dfy_ssd_evaluation_result_log)
            print('类别为2 预测为：卡车的box个数：', len(results[2]), results[2][:10], file=self.dfy_ssd_evaluation_result_log)
            print('类别为3 预测为：行人的box个数：', len(results[3]), results[3][:10], file=self.dfy_ssd_evaluation_result_log)
            print('类别为4 预测为：自行车的box个数：', len(results[4]), results[4][:10], file=self.dfy_ssd_evaluation_result_log)
            print('类别为5 预测为：交通灯的box个数：', len(results[5]), results[5][:10], file=self.dfy_ssd_evaluation_result_log)

        if ret:
            return results

    def write_predictions_to_txt(self,
                                 classes=None,
                                 out_file_prefix='comp3_det_test_',
                                 verbose=True):
        '''
        Writes the predictions for all classes to separate text files according to the Pascal VOC results format.

        Arguments:
            classes (list, optional): `None` or a list of strings containing the class names of all classes in the dataset,
                including some arbitrary name for the background class. This list will be used to name the output text files.
                The ordering of the names in the list represents the ordering of the classes as they are predicted by the model,
                i.e. the element with index 3 in this list should correspond to the class with class ID 3 in the model's predictions.
                If `None`, the output text files will be named by their class IDs.
            out_file_prefix (str, optional): A prefix for the output text file names. The suffix to each output text file name will
                be the respective class name followed by the `.txt` file extension. This string is also how you specify the directory
                in which the results are to be saved.
            verbose (bool, optional): If `True`, will print out the progress during runtime.

        Returns:
            None.
        '''

        if self.prediction_results is None:
            raise ValueError(
                "There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        # We generate a separate results file for each class.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Writing results file for class {}/{}.".format(class_id, self.n_classes))

            if classes is None:
                class_suffix = '{:04d}'.format(class_id)
            else:
                class_suffix = classes[class_id]

            results_file = open('{}{}.txt'.format(out_file_prefix, class_suffix), 'w')

            for prediction in self.prediction_results[class_id]:
                prediction_list = list(prediction)
                prediction_list[0] = '{:06d}'.format(int(prediction_list[0]))
                prediction_list[1] = round(prediction_list[1], 4)
                prediction_txt = ' '.join(map(str, prediction_list)) + '\n'
                results_file.write(prediction_txt)

            results_file.close()

        if verbose:
            print("All results files saved.")

    def get_num_gt_per_class(self,
                             ignore_neutral_boxes=True,
                             verbose=True,
                             ret=False):
        '''
        获取每个类的人工标注框的总数。

        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `True`, only non-neutral ground truth boxes will be counted, otherwise all ground truth boxes will
                be counted.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the list of counts.

        Returns:
            None by default. Optionally, a list containing a count of the number of ground truth boxes for each class across the
            entire dataset.
        '''

        if self.data_generator.labels is None:
            raise ValueError(
                "Computing the number of ground truth boxes per class not possible, no ground truth given.")

        num_gt_per_class = np.zeros(shape=(self.n_classes + 1), dtype=np.int)

        class_id_index = self.gt_format['class_id']

        ground_truth = self.data_generator.labels

        if verbose:
            print('Computing the number of positive ground truth boxes per class.')
            tr = trange(len(ground_truth), file=sys.stdout)
        else:
            tr = range(len(ground_truth))

        # 遍历数据集中所有图像的人工标注框，即一张图片一张图片的遍历
        for i in tr:

            boxes = np.asarray(ground_truth[i]) # np.asarray的作用类似于np.array(),只是参数少于np.array()

            # 遍历当前图像的所有人工标注框
            for j in range(boxes.shape[0]):

                if ignore_neutral_boxes and not (self.data_generator.eval_neutral is None):
                    if not self.data_generator.eval_neutral[i][j]:
                        # If this box is not supposed to be evaluation-neutral,
                        # increment the counter for the respective class ID.
                        class_id = boxes[j, class_id_index]
                        num_gt_per_class[class_id] += 1
                else:
                    # If there is no such thing as evaluation-neutral boxes for our dataset, always increment the counter for the respective class ID.
                    class_id = boxes[j, class_id_index]
                    num_gt_per_class[class_id] += 1

        self.num_gt_per_class = num_gt_per_class  #获取每个类的人工标注框的总数，并将每类人工标注框的个数存储在mum_gt_per_class,形成一个列表

        if ret:
            return num_gt_per_class

    def match_predictions(self,
                          ignore_neutral_boxes=True,
                          matching_iou_threshold=0.5,
                          border_pixels='include',
                          sorting_algorithm='quicksort',
                          verbose=True,
                          ret=False):
        '''
        Matches predictions to ground truth boxes.

        Note that `predict_on_dataset()` must be called before calling this method.

        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `False`, even boxes that are annotated as neutral will be counted into the evaluation. If `True`,
                neutral boxes will be ignored for the evaluation. An example for evaluation-neutrality are the ground truth boxes
                annotated as "difficult" in the Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the true and false positives.

        Returns:
            默认情况下没有。可选地，包含每个类的真阳性、假阳性、累积真阳性和累积假阳性的四个嵌套列表。
        '''

        if self.data_generator.labels is None:
            # 以单个图片为单位，存放其 labels 人工标注框 类别和box坐标
            raise ValueError("Matching predictions to ground truth boxes not possible, no ground truth given.")

        if self.prediction_results is None:
            raise ValueError(
                "There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        class_id_gt = self.gt_format['class_id']
        xmin_gt = self.gt_format['xmin']
        ymin_gt = self.gt_format['ymin']
        xmax_gt = self.gt_format['xmax']
        ymax_gt = self.gt_format['ymax']

        # 将人工标注框转换为我们需要做的更有效的格式，即通过图像ID反复访问ground truth.
        ground_truth = {}   # dict key:图片id  value:图片的人工标注gt box（一个或者多个值）
        eval_neutral_available = not (self.data_generator.eval_neutral is None)  # 我们是否有注解来决定人工标注框是否应该是中立的。

        if self.dfy_ssd_evaluation_result_log:
            print('\n开始匹配预测的所有box，按类别来，判断每个类别预测的所有box是否正确？首先看类别 其次看iou', file=self.dfy_ssd_evaluation_result_log)
            print('在match_predictions函数中：准备填充字典ground_truth,key:image_id,value:图片的all人工标注框,共有%d张图片' % len(self.data_generator.image_ids), file=self.dfy_ssd_evaluation_result_log)

        for i in range(len(self.data_generator.image_ids)):
            # 100 loop  一张图片 一张图片的来 把字典ground_truth填充满了
            image_id = str(self.data_generator.image_ids[i])
            labels = self.data_generator.labels[i]
            if ignore_neutral_boxes and eval_neutral_available:
                ground_truth[image_id] = (np.asarray(labels), np.asarray(self.data_generator.eval_neutral[i]))
            else:
                # 我运行了是：走这个条件  没有评估中立样本
                ground_truth[image_id] = np.asarray(labels)

        true_positives = [[]]  # The false positives for each class, sorted by descending confidence.
        false_positives = [[]]  # The true positives for each class, sorted by descending confidence.
        cumulative_true_positives = [[]]
        cumulative_false_positives = [[]]

        # 遍历所有类。
        for class_id in range(1, self.n_classes + 1):
            # 按类别的 一个一个看预测的结果咋样？

            predictions = self.prediction_results[class_id] # 找出一个类的所有预测anchor box

            # 将匹配结果存储在这些列表中:
            true_pos = np.zeros(len(predictions),
                                dtype=np.int)  # 1 for every prediction that is a true positive, 0 otherwise
            false_pos = np.zeros(len(predictions),
                                 dtype=np.int)  # 1 for every prediction that is a false positive, 0 otherwise

            # 如果这里没有任何预测anchor box，我们就在这里结束。
            if len(predictions) == 0:
                if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                    print("在全部的评估数据集中，预测结果为这个类别id:{}的box数目为0，共总的正样本类别数为：{} 结束本次循环，看下一个类别的".format(class_id, self.n_classes),
                          file=self.dfy_ssd_evaluation_result_log)
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                continue

            # 将该类的预测列表转换为结构化数组，以便我们能够根据置信度对其进行排序。
            # 获取在结构化数组中存储图像ID字符串所需的字符数。
            # assert num_chars_per_image_id == 19 + 6, 其中19是照片名的长度
            num_chars_per_image_id = len(str(predictions[0][0])) + 6  # 保留一些字符缓冲区，以防某些图像id比其他id长。

            # 为结构化数组创建数据类型。
            preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])
            # 创建结构化数组
            predictions = np.array(predictions, dtype=preds_data_type)
            if self.dfy_ssd_evaluation_result_log:
                print('类别为：%d的所有预测结果的shape(还是一维向量)：%s' % (class_id, predictions.shape), file=self.dfy_ssd_evaluation_result_log)
                if class_id == 5:
                    print('交通灯class_id == 5类别的所有预测值：', len(predictions), predictions, file=self.dfy_ssd_evaluation_result_log)

            #  根据置信度从大到小排序
            descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm) # 函数返回的是数组值从大到小的索引值。
            predictions_sorted = predictions[descending_indices]
            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                print('交通灯class_id == 5类别的所有预测box(按置信度从大到小排序后的)：', predictions_sorted, file=self.dfy_ssd_evaluation_result_log)

            if verbose:
                # verbose = True 走这个条件
                tr = trange(len(predictions), file=sys.stdout)
                tr.set_description(
                    "Matching predictions按类别匹配预测的box to ground truth, class {}/{}.".format(class_id, self.n_classes))
            else:
                tr = range(len(predictions.shape))

            # 跟踪哪些人工标注框已经匹配到一个检测。
            gt_matched = {}  # dict key:image_id  value: bool列表 每个位置代表当前图片的人工标注框是否已经被匹配了

            # 遍历所有预测anchor box
            for i in tr:
                # 具体某个类别里 一个预测box 一个预测box的来
                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']]))  # 将结构化数组元素转换为常规数组。

                # 获取该预测的相关人工标注框，即与预测的图像ID和类ID匹配的所有人工标注框。
                # 人工标注框可以是一个带有' (ground_truth_boxes, eval_neutral_boxes) '的元组。或者只是“ground_truth_boxes”。
                if ignore_neutral_boxes and eval_neutral_available:
                    gt, eval_neutral = ground_truth[image_id]
                else:
                    # print('东方耀，我走这个条件')
                    # ground_truth是字典dict  key:图片id  value:图片的人工标注gt box
                    gt = ground_truth[image_id]
                gt = np.asarray(gt)  # gt 是某个图片的 所有人工标注框
                class_mask = gt[:, class_id_gt] == class_id
                gt = gt[class_mask]
                if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                    # 预测的一个box 与 该图片中该类别的所有人工标注的box
                    print('\n交通灯class_id == 5类别,排序后：(第%d个)预测的一个box与该图片image_id=%s中这个类别(交通灯)的所有人工标注框：' % (i, image_id),
                          gt.shape, gt, file=self.dfy_ssd_evaluation_result_log)
                if ignore_neutral_boxes and eval_neutral_available:
                    eval_neutral = eval_neutral[class_mask]

                if gt.size == 0:
                    if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                        print('类别class_id=%d当中预测的第%d个box在它对应的图片中所有该类别的人工标注框为0，说明预测的这个box类别肯定错误啊！' % (class_id, i),
                              file=self.dfy_ssd_evaluation_result_log)
                    # 如果图像不包含此类的任何对象，则预测为假阳性。
                    false_pos[i] = 1
                    continue

                # 用同一类的所有人工标注框计算该预测的iou。
                overlaps = iou(boxes1=gt[:, [xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               coords='corners',
                               mode='element-wise',
                               border_pixels=border_pixels)
                if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                    print('交通灯class_id == 5类别,(第%d个)预测的一个box与该图片image_id=%s中这个类别(交通灯)的所有人工标注框_计算iou：' % (i, image_id),
                          overlaps.shape, overlaps, file=self.dfy_ssd_evaluation_result_log)

                # 对于每次检测，匹配重叠最高的人工标注框。同一个人工标注框有可能被匹配到多个探测。
                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]
                if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                    print('取iou的最大值(所有人工标注的框中与预测的box重合度最大的那个)(看是否大于设定的iou匹配阈值)：', gt_match_overlap, '匹配的索引为：', gt_match_index,
                          file=self.dfy_ssd_evaluation_result_log)

                if gt_match_overlap < matching_iou_threshold:
                    if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                        print('类别class_id=%d当中预测的第%d个box跟其对应的所有人工标注框iou的最大值 还要小于 我们设定的iou匹配阈值:标识为预测错误的box：第%d个' % (class_id, i, i),
                              file=self.dfy_ssd_evaluation_result_log)
                    # 假阳性，违反IoU阈值:那些匹配重叠低于阈值的预测将成为假阳性。
                    false_pos[i] = 1
                else:
                    if not (ignore_neutral_boxes and eval_neutral_available) or (eval_neutral[gt_match_index] == False):
                        # 如果这不是一个基本事实，那就应该是评估中立的 (例如，应该跳过评估)或者我们甚至没有中立框的概念。
                        if not (image_id in gt_matched):
                            # 真正:如果这个预测匹配的人工标注框没有匹配到一个不同的预测，我们有一个真正。
                            true_pos[i] = 1
                            gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                            gt_matched[image_id][gt_match_index] = True
                            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                                print('交通灯类别中：标识为正确预测的box：第%d个' % i, file=self.dfy_ssd_evaluation_result_log)
                                print('交通灯类别中：该预测box对应image_id=%s中所有人工标注框:gt=' % image_id, gt.shape, file=self.dfy_ssd_evaluation_result_log)
                                print('该预测box匹配上(占用)的人工标注框索引：gt_match_index:', gt_match_index, file=self.dfy_ssd_evaluation_result_log)
                                print('第一种情况：交通灯类别中：image_id=%s没在gt_matched中的情况：之后gt_matched=' % image_id, gt_matched, file=self.dfy_ssd_evaluation_result_log)
                        elif not gt_matched[image_id][gt_match_index]:
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                                print('交通灯类别中：标识为正确预测的box：第%d个' % i, file=self.dfy_ssd_evaluation_result_log)
                                print('第二种情况(该图片image_id=%s,预测的第%d个box由前面计算的iou最大值而匹配上的人工标注框索引gt_match_index=%d还没有被其他预测box占用)：, gt_matched=' % (image_id, i, gt_match_index),
                                      gt_matched, file=self.dfy_ssd_evaluation_result_log)
                            true_pos[i] = 1
                            gt_matched[image_id][gt_match_index] = True
                            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                                print('第二种情况,gt_matched(字典)更新之后是：', gt_matched, file=self.dfy_ssd_evaluation_result_log)

                        else:
                            # False positive, duplicate detection:
                            # If the matched ground truth box for this prediction has already been matched
                            # to a different prediction previously, it is a duplicate detection for an
                            # already detected object, which counts as a false positive.
                            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                                print('规定：再次匹配到同一个gt box是错误的，先匹配到的conf是要大些的(已经排序了)', file=self.dfy_ssd_evaluation_result_log)
                                print('第三种情况(该图片image_id=%s中,前面计算的最大iou匹配上的人工标注框已经被其他预测box占用)：交通灯类别中：标识为错误预测的box：第%d个' % (image_id, i),
                                      file=self.dfy_ssd_evaluation_result_log)
                            false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)
            # cum sum 累计求和
            cumulative_true_pos = np.cumsum(true_pos)  # Cumulative sums of the true positives
            cumulative_false_pos = np.cumsum(false_pos)  # Cumulative sums of the false positives
            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                print('\n交通灯class_id == 5类别(共%d个预测的box),预测正确的box:true_pos=' % len(predictions), true_pos, file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别(共%d个预测的box),预测错误的box:false_pos=' % len(predictions), false_pos, file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别(共%d个预测的box),cumulative_true_pos=' % len(predictions), cumulative_true_pos, file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别(共%d个预测的box),cumulative_false_pos=' % len(predictions), cumulative_false_pos, file=self.dfy_ssd_evaluation_result_log)

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        print('背景class_id ==0类别的：', true_positives[0], false_positives[0], cumulative_true_positives[0], cumulative_false_positives[0])
        print('汽车class_id ==1类别的：(共%d个预测的box),预测正确的box:true_pos=' % len(false_positives[1]), 'true_positives[1]太长不显示了')
        print('卡车class_id ==2类别的：(共%d个预测的box),预测正确的box:true_pos=' % len(false_positives[2]), 'true_positives[2]')
        print('行人class_id ==3类别的：(共%d个预测的box),预测正确的box:true_pos=' % len(cumulative_true_positives[3]), 'true_positives[3]')
        print('自行车class_id ==4类别的：(共%d个预测的box),预测正确的box:true_pos=' % len(cumulative_false_positives[4]), 'true_positives[4]')
        print('交通灯class_id == 5类别(共%d个预测的box),预测正确的box:true_pos=' % len(true_positives[5]), 'true_positives[5]')
        print('\n交通灯class_id == 5类别(共%d个预测的box),预测错误的box:false_pos=' % len(false_positives[5]), 'false_positives[5]')
        print('交通灯class_id == 5类别,cumulative_true_pos=', 'cumulative_true_positives[5]')
        print('交通灯class_id == 5类别,cumulative_false_pos=', 'cumulative_false_positives[5]\n')
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.cumulative_true_positives = cumulative_true_positives
        self.cumulative_false_positives = cumulative_false_positives

        if ret:
            return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives

    def compute_precision_recall(self, verbose=True, ret=False):
        '''
        计算所有类的精准率和召回率。

        Note that `match_predictions()` must be called before calling this method.

        Arguments:
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the precisions and recalls.

        Returns:
            默认情况下没有。可选地，两个嵌套列表包含每个类的累积精度和召回
        '''

        if (self.cumulative_true_positives is None) or (self.cumulative_false_positives is None):
            raise ValueError(
                "True and false positives not available. You must run `match_predictions()` before you call this method.")

        if (self.num_gt_per_class is None):
            raise ValueError(
                "Number of ground truth boxes per class not available. You must run `get_num_gt_per_class()` before you call this method.")

        cumulative_precisions = [[]]
        # 类别为0的 背景 就是 []
        cumulative_recalls = [[]]

        # 遍历所有类。
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("计算精准率和召回率, class正样本类别id： {}/{}".format(class_id, self.n_classes))

            # 为什么用cumsum累加的？
            # tp:累计的预测正确的box个数(顺序是：预测box的置信度从大到小的，如果调整预测结果解码时的置信度阈值将直接影响该类预测box的总数量)
            tp = self.cumulative_true_positives[class_id]
            fp = self.cumulative_false_positives[class_id]
            # tp+fp:随着预测的box的增加 tp正确的个数 fp就是错误的个数
            # 交通灯类别的tp+fp=[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18]
            # tp + fp > 0 说明该类预测的box总个数是大于0的 精准率=针对预测的结果的,在预测的总box个数中到底有几个是预测正确的？
            # cumulative_precision：随着预测的box的增加，该类别 精准率的变化情况
            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0)  # 1D array with shape `(num_predictions,)`
            # 召回率：针对实际人工标注结果的，在该类别所有人工标注框中到底有多少个是被正确预测出来的？
            # cumulative_recall: 随着预测的box的增加，该类别 召回率的变化情况 肯定是越来越大的
            cumulative_recall = tp / self.num_gt_per_class[class_id]  # 1D array with shape `(num_predictions,)`

            if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                print('\n计算精准率和召回率(按类别)：交通灯class_id == 5类别(共%d个预测的box)' % len(tp), file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别,累计的预测正确的box(顺序是：预测box的置信度从大到小的，如果调整预测结果解码时的置信度阈值将直接影响该类预测box的总数量)：',
                      tp, file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别,累计的预测错误的box:', fp, file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别,随着预测box的总数量的增加，该类别 精准率的变化情况cumulative_precision：', len(cumulative_precision), cumulative_precision, file=self.dfy_ssd_evaluation_result_log)
                print('交通灯class_id == 5类别,随着预测box的总数量的增加，该类别 召回率的变化情况(肯定是越来越大的)cumulative_recall：', len(cumulative_recall), cumulative_recall, file=self.dfy_ssd_evaluation_result_log)
                print('计算精准率和召回率的最终结果：按类别index放在列表self.cumulative_precisions与self.cumulative_recalls中', file=self.dfy_ssd_evaluation_result_log)

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        self.cumulative_precisions = cumulative_precisions
        self.cumulative_recalls = cumulative_recalls

        if ret:
            return cumulative_precisions, cumulative_recalls

    def compute_average_precisions(self, mode='sample', num_recall_points=11, verbose=True, ret=False):
        '''
        计算每个类的平均精度。

        Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
        and post-2010 (integration) algorithm versions.

        Note that `compute_precision_recall()` must be called before calling this method.

        Arguments:
            mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision will be computed
                according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled for `num_recall_points`
                recall values. In the case of 'integrate', the average precision will be computed according to the Pascal VOC formula that
                was used from VOC 2010 onward, where the average precision will be computed by numerically integrating over the whole
                preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just the limit case
                of 'sample' mode as the number of sample points increases. For details, see the references below.
            num_recall_points (int, optional): Only relevant if mode is 'sample'. The number of points to sample from the precision-recall-curve
                to compute the average precisions. In other words, this is the number of equidistant recall values for which the resulting
                precision will be computed. 11 points is the value used in the official Pascal VOC pre-2010 detection evaluation algorithm.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the average precisions.

        Returns:
            None by default. Optionally, a list containing average precision for each class.

        References:
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
        '''

        if (self.cumulative_precisions is None) or (self.cumulative_recalls is None):
            raise ValueError(
                "Precisions and recalls not available. You must run `compute_precision_recall()` before you call this method.")

        if not (mode in {'sample', 'integrate'}):
            raise ValueError("`mode` can be either 'sample' or 'integrate', but received '{}'".format(mode))
        # 类别为背景的 平均精准率
        average_precisions = [0.0]

        # 遍历所有类。
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("按类别来计算平均精准率, class {}/{}".format(class_id, self.n_classes))

            cumulative_precision = self.cumulative_precisions[class_id]
            cumulative_recall = self.cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):
                    # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]

                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        # 直接选一个最大的精准率
                        precision = np.amax(cum_prec_recall_greater_t)
                    if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                        print('\n计算average精准率(按类别)交通灯class_id == 5类别：循环num_recall_points=%d次,本次的累计召回率阈值t=%0.2f,本次得出的精准率=%0.5f' % (num_recall_points, t, precision),
                              file=self.dfy_ssd_evaluation_result_log)

                    average_precision += precision

                average_precision /= num_recall_points
                if self.dfy_ssd_evaluation_result_log and (class_id == 5):
                    print('\n结果：交通灯class_id == 5类别, average精准率mode=%s,召回率点数=%d,平均精准率(上面每次得出的精准率相加除以总次数)：%.5f' % (mode, num_recall_points, average_precision),
                          file=self.dfy_ssd_evaluation_result_log)
                    print('平均精准率的最终结果：按类别index放在列表self.average_precisions中', file=self.dfy_ssd_evaluation_result_log)

            elif mode == 'integrate':

                # We will compute the precision at all unique recall values.
                unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall,
                                                                                        return_index=True,
                                                                                        return_counts=True)

                # Store the maximal precision for each recall value and the absolute difference
                # between any two unique recal values in the lists below. The products of these
                # two nummbers constitute the rectangular areas whose sum will be our numerical
                # integral.
                maximal_precisions = np.zeros_like(unique_recalls)
                recall_deltas = np.zeros_like(unique_recalls)

                # Iterate over all unique recall values in reverse order. This saves a lot of computation:
                # For each unique recall value `r`, we want to get the maximal precision value obtained
                # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
                # values after a given iteration, then in the next iteration, in order compute the maximal
                # precisions for the last `l > k` recall values, we only need to compute the maximal precision
                # for `l - k` recall values and then take the maximum between that and the previously computed
                # maximum instead of computing the maximum over all `l` values.
                # We skip the very last recall value, since the precision after between the last recall value
                # recall 1.0 is defined to be zero.
                for i in range(len(unique_recalls) - 2, -1, -1):
                    begin = unique_recall_indices[i]
                    end = unique_recall_indices[i + 1]
                    # When computing the maximal precisions, use the maximum of the previous iteration to
                    # avoid unnecessary repeated computation over the same precision values.
                    # The maximal precisions are the heights of the rectangle areas of our integral under
                    # the precision-recall curve.
                    maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]),
                                                       maximal_precisions[i + 1])
                    # The differences between two adjacent recall values are the widths of our rectangle areas.
                    recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        self.average_precisions = average_precisions

        if ret:
            return average_precisions

    def compute_mean_average_precision(self, ret=True):
        '''
        Computes the mean average precision over all classes.

        Note that `compute_average_precisions()` must be called before calling this method.

        Arguments:
            ret (bool, optional): If `True`, returns the mean average precision.

        Returns:
            A float, the mean average precision, by default. Optionally, None.
        '''

        if self.average_precisions is None:
            raise ValueError(
                "Average precisions not available. You must run `compute_average_precisions()` before you call this method.")

        mean_average_precision = np.average(
            self.average_precisions[1:])  # The first element is for the background class, so skip it.
        self.mean_average_precision = mean_average_precision

        if ret:
            return mean_average_precision
