
import numpy as np
from keras.utils import np_utils
n_classes = 5 + 1

"""
总结：SSDLoss损失函数的定义
ssd_loss = SSDLoss(neg_pos_ratio=3,
                   n_neg_min=1,
                   alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

SSD的损失：分类损失-log(实际类别对应的预测的概率值p)
交叉熵体现了模型输出的概率分布和真实样本的概率分布的相似程度
定位损失smooth_L1_loss
负样本：实际值中 类别为背景的边界框box
正样本：实际值中 类别为具体物体类别的边界框box
batch_size 表示多少张图片 每张图片中有n_boxes_total个样本(正样本 负样本 中立样本)
定位损失：只考虑每张图片中正样本的损失之和
分类损失：分为正样本的分类损失 + 部分数量的负样本的分类损失之和
限定负样本的数量（以batch_size张图片为单位的，而不是一张图片中满足条件）：最多是正样本数量的neg_pos_ratio=3倍 最少是self.n_neg_min
如果在batch_size张图片中没有一个正样本，导致选取的负样本个数也为0，因为负样本数量最多是正样本数量的neg_pos_ratio=3倍，此时总的损失为0，导致模型无法学习
此时n_neg_min这个参数就开始起作用了，这个值确保每一次迭代梯度都有足够的负样本参与, 即便正样本的数量很小, 或者是0, 也想要使用负样本进行训练.
在batch_size张图片中只取负样本的分类损失前面最大的几个（控制负样本的数量）
体现了难例挖掘思想：找到某些负样本 其分类损失最大的 才是难例
alpha 越大 更多的考虑 正样本的定位损失
"""


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


y_true = np.full(shape=(2, 3, 18), fill_value=-1, dtype=np.float)
y_pred = np.full(shape=(2, 3, 18), fill_value=-1, dtype=np.float)
# 0 表示 负样本
# 1-5 表示 正样本
part1_dim_6_true = np_utils.to_categorical(y=[0, 0, 0, 5, 0, 0], num_classes=n_classes).reshape((2, 3, 6))
# print(part1_dim_6_true)  # y=[0, 0, 0, 5, 0, 0]:表示有5个元素，第1，2，3，5，6个元素都属于第一个类别，第4个元素属于第六个类别
y_true[:, :, :6] = part1_dim_6_true
y_true[:, :, 6] = 16
y_true[:, :, 7] = 17
y_true[:, :, 8] = 18
y_true[:, :, 9] = 19

y_true[:, :, 10] = 26
y_true[:, :, 11] = 27
y_true[:, :, 12] = 28
y_true[:, :, 13] = 29

y_true[:, :, 14] = 36
y_true[:, :, 15] = 37
y_true[:, :, 16] = 38
y_true[:, :, 17] = 39
print(y_true)
print('y_true.shape=', y_true.shape)


part1_dim_6_pred = np.full((2, 3, 6), fill_value=888, dtype=np.float)
part1_dim_6_pred[0, 0, :] = [0.00426978, 0.01160646, 0.08576079, 0.23312201, 0.03154963, 0.63369132]
part1_dim_6_pred[0, 1, :] = [0.03163502, 0.00157501, 0.63540629, 0.08599289, 0.01163787, 0.23375291]
part1_dim_6_pred[0, 2, :] = [6.02561205e-04, 1.63793117e-03, 2.43090540e-01, 4.45235855e-03, 6.60788597e-01, 8.94280120e-02]
part1_dim_6_pred[1, 0, :] = [2.67614720e-01, 6.63350570e-04, 1.80317380e-03, 6.63350570e-04, 1.80317380e-03, 7.27452231e-01]
part1_dim_6_pred[1, 1, :] = [2.78934145e-10, 6.91408620e-13, 9.99954601e-01, 4.53978687e-05, 1.87944349e-12, 7.58221619e-10]
part1_dim_6_pred[1, 2, :] = [4.53978686e-05, 2.06106005e-09, 9.99954600e-01, 9.35719813e-14, 8.53266023e-17, 3.44232082e-14]
y_pred[:, :, :6] = part1_dim_6_pred
y_pred[:, :, 6] = 15
y_pred[:, :, 7] = 19
y_pred[:, :, 8] = 8
y_pred[:, :, 9] = 16

y_pred[:, :, 10] = 22
y_pred[:, :, 11] = 28
y_pred[:, :, 12] = 25
y_pred[:, :, 13] = 33

y_pred[:, :, 14] = 34
y_pred[:, :, 15] = 36
y_pred[:, :, 16] = 40
y_pred[:, :, 17] = 42
print(y_pred)
print('y_pred.shape=', y_pred.shape)

# 已经数据终于构造完毕


def smooth_L1_loss(y_true, y_pred):
    '''
    计算 smooth L1 loss.

    Arguments:
        y_true (nD tensor): 包含人工标准的样本的值, 形状为 `(batch_size, #boxes, 4)`. 最后维度
            包含如下四个坐标值 `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): 包含预测的数据, 和 `y_true` 的形状一样.

    Returns:
        返回smooth L1 loss, 2维 tensor, 形状 (batch, n_boxes_total).

    References:
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = np.abs(y_true - y_pred)
    # print('absolute_loss:', absolute_loss)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    # print('square_loss:', square_loss)
    l1_loss = np.where(np.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    # 在最后一个维度求和 高维数据 具体的值落在最后一维
    # return np.reduce_sum(l1_loss, axis=-1)
    # print('l1_loss:', l1_loss)
    return np.sum(l1_loss, axis=-1)


print(y_true[:, :, -12:-8])
print(y_pred[:, :, -12:-8])
print('smooth_L1_loss=', smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))


def log_loss(y_true, y_pred):
    '''
    计算 softmax log loss.

    Arguments:
        y_true (nD tensor): 人工标注的值, 形状 (batch_size, #boxes, #classes)
        y_pred (nD tensor): 预测的值, 与 `y_true` 形状一样.

    Returns:
        返回 softmax log loss, 2 维 tensor, 形状为 (batch, n_boxes_total).
    '''
    # 确保 `y_pred` 不包含为0的值
    y_pred = np.maximum(y_pred, 1e-15)
    # 计算 log loss  查看分类的loss
    log_loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    # log_loss = categorical_crossentropy(y_true, y_pred)
    return log_loss


print(y_true[:, :, :-12])
print(y_pred[:, :, :-12])
print('log_loss=', log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))


def compute_loss_dfy(y_true, y_pred):
    neg_pos_ratio = 3.0
    n_neg_min = 1.0
    alpha = 1.0
    # 输出类型: np.int32
    batch_size = y_pred.shape[0]
    # 输出类型: np.int32, `n_boxes` 表达每个图像对应的所有边界框的个数, 不是特征图的每个位置对应的边界框的数量
    n_boxes = y_pred.shape[1]
    # 有多少张图片？
    assert batch_size == 2
    # 每张图片有多少个边界框？
    assert n_boxes == 3

    # 1: 为每张图片的每个bounding box边界框（不分正负样本）计算分类和定位loss
    classification_loss = log_loss(y_true[:, :, :-12], y_pred[:, :, :-12])  # 输出形状: (batch_size, n_boxes_total)
    print('classification_loss:', classification_loss)
    localization_loss = smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8])  # 输出形状: (batch_size, n_boxes_total)
    print('localization_loss:', localization_loss)

    # 2: 计算正负样本的分类 losses

    # 创建正样本, 负样本的 mask.(ssd中样本不是指图片 而是边界框)
    # 6个类别：0,1,2,3,4,5  0为背景 y_true里类别是one-hot编码的
    # 在y_true的第0个维度 表示背景 后面的1-5个维度表示 具体物体的类别（正样本的）
    # positives_mask 在后面5个维度上取最大值 不是0就是1  如果是0 则表示肯定在0的维度上是1（背景 负样本） 与negatives相反
    negatives_mask = y_true[:, :, 0]  # 形状 (batch_size, n_boxes)
    print('原始的负样本mask:', negatives_mask)
    positives_mask = np.max(y_true[:, :, 1:-12], axis=-1)  # 形状 (batch_size, n_boxes)
    print('正样本的mask:', positives_mask)

    # 在实际值y_true中batch_size张图片 所有正样本（边界框）(类别标签 1 到 n)的数量
    n_positive = np.sum(positives_mask)
    print('n_positive（正样本个数）:', n_positive)

    # 不考虑负样本的情况
    # (Keras 计算 loss 的时候只计算每一幅图的损失, 而不是整个 batch 的损失之和, 所以我们需要求和).
    # 计算每张图片中 所有正样本（边界框的类别为1-5）的分类损失之和
    pos_class_loss = np.sum(classification_loss * positives_mask, axis=-1)  # 得到每张图片的正样本损失 # Tensor of shape (batch_size,)
    print('正样本的分类损失之和(每张图片中)pos_class_loss:', pos_class_loss)

    # 得到每张图片当中 所有负样本（边界框的类别为0）的分类损失 这里并没有求和
    neg_class_loss_all = classification_loss * negatives_mask  # Tensor of shape (batch_size, n_boxes)
    print('neg_class_loss_all:', neg_class_loss_all)

    # 计算不为0的数目：这里实际就是 所有batch_size张图片中 所有负样本（边界框的类别为0）的数量
    n_negative = np.count_nonzero(neg_class_loss_all)  # The number of non-zero loss entries in `neg_class_loss_all`
    # n_negative = np.sum(negatives_mask) 这里注意负样本的数量 为何不直接用负样本的掩码求和 有可能某个负样本的分类损失就是0的情况
    # 思想：难例挖掘：找到某些负样本 其分类损失最大的 才是难例
    print('n_negative（负样本个数）:', n_negative)

    # 计算我们需要处理的负样本的数量. 最多保留 `self.neg_pos_ratio` 乘以 `y_true` 中正样本的数量, 但是至少 `self.n_neg_min` 个.
    # 限定负样本的数量（以batch_size张图片为单位的，而不是一张图片中满足这样）：最多是正样本数量的3倍 最少是self.n_neg_min
    n_negative_keep = np.minimum(np.maximum(neg_pos_ratio * np.int(n_positive), n_neg_min),
                                 n_negative)
    print('n_negative_keep（只保留的负样本个数，正负样本的平衡）:', n_negative_keep)

    print('下面开始求 每张图片中所有负样本的分类损失之和（难点）')
    neg_class_loss_all_1D = np.reshape(neg_class_loss_all, (-1, ))
    print(neg_class_loss_all_1D)

    # 在batch_size张图片中只取所有负样本的分类损失 前面最大的几个 （控制负样本的数量）
    indices = np.argsort(-neg_class_loss_all_1D)[:np.int(n_negative_keep)]
    print(indices)
    negatives_keep_mask = np.zeros(shape=neg_class_loss_all_1D.shape)
    for i in indices:
        negatives_keep_mask[i] = 1
    negatives_keep_mask = negatives_keep_mask.reshape((batch_size, n_boxes))
    print('保留的负样本的mask(会丢掉一部分负样本):\n', negatives_keep_mask)
    neg_class_loss = np.sum(classification_loss * negatives_keep_mask, axis=-1)
    print('负样本的分类损失之和(每张图片中)neg_class_loss：', neg_class_loss)
    print('end.... 每张图片中所有负样本的分类损失之和（难点）')

    # 分类损失=正样本的分类损失 + 负样本的分类损失
    # 负样本的分类损失 不能这样算   1、因为负样本太多了，需要限制个数
    # neg_class_loss = np.sum(classification_loss * negatives_mask, axis=-1)  # Tensor of shape (batch_size,)
    class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)
    print('总的分类损失(每张图片中)class_loss：', class_loss)
    print('总的分类损失(每张图片中)(错误class_loss)：', np.sum(classification_loss, axis=-1))

    # 3: Compute the localization loss for the positive targets.
    #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).
    # 只考虑正样本的定位损失
    # 计算每张图片中 所有正样本（边界框的类别为1-5）的定位损失之和
    loc_loss = np.sum(localization_loss * positives_mask, axis=-1)  # Tensor of shape (batch_size,)
    print('只考虑正样本的定位损失：', loc_loss)

    # 4: Compute the total loss.
    # 这里为何要除以正样本的个数？
    # total_loss = (class_loss + alpha * loc_loss) / n_positive
    # 下面这种改进 只是怕n_positive == 0而变的
    total_loss = (class_loss + alpha * loc_loss) / np.maximum(1.0, n_positive)  # In case `n_positive == 0`
    print('total_loss:', total_loss)

    #Keras有一个烦人的习惯，那就是将损失除以批处理大小。
    # 在我们的例子中，这很糟糕，因为平均损失的相关标准是批处理中的正盒数(我们在上面的行中除以它)，而不是批处理大小。
    # 所以为了恢复Keras的平均批大小，我们必须乘以它。
    total_loss = total_loss * batch_size
    # total_loss = np.sum(total_loss, axis=-1)

    return total_loss


print('实际值：', y_true)
print('实际值 shape：', y_true.shape)
print('\n')
print('预测值：', y_pred)
print('预测值 shape：', y_pred.shape)
print('\n')
print('compute_loss=', compute_loss_dfy(y_true, y_pred))

