'''
SSD的损失函数的 Keras 实现, 只支持TensorFlow.
'''

import tensorflow as tf

class SSDLoss:
    '''
    SSD损失函数, 参考 https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): 正负样本比值，参与损失函数值计算的背景和真实目标的个数比例的最大值.
                真实的训练数据是没有人工标准的背景, 但是我们的`y_true`中包含了被认为是背景的Anchor.
                `y_true` 中被认为是背景的框的数量远远大于人工标注的正样本的数量. 需要做一些筛选.
                默认取值为 3.
            n_neg_min (int, optional): 每一批图像中最为负样本的背景的边界框的数量的最小值. 这个值
                确保每一次迭代梯度都有足够的负样本参与, 即便正样本的数量很小, 或者是0, 也想要使用负样
                本进行训练.
            alpha (float, optional): 用于平衡定位误差在总的误差计算中占的比重. 默认值为0.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
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
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)  # tf.less返回了两个张量各元素比较(x<y)得到的真假值组成的张量
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        计算 softmax log loss.

        Arguments:
            y_true (nD tensor): 人工标注的值, 形状 (batch_size, #boxes, #classes)
            y_pred (nD tensor): 预测的值, 与 `y_true` 形状一样.

        Returns:
            返回 softmax log loss, 2 维 tensor, 形状为 (batch, n_boxes_total).
        '''
        # 确保 `y_pred` 不包含为0的值
        y_pred = tf.maximum(y_pred, 1e-15)  # 用法tf.maximum(a,b),返回的是a,b之间的最大值
        # 计算 log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)  # tf.reduce_sum() 用于计算张量tensor沿着某一维度的和
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        计算 SSD 模型的 loss

        Arguments:
            y_true (array): Numpy 数组, 形状 `(batch_size, #boxes, #classes + 12)`,
                其中 `#boxes` 为模型为每一幅图预测的边界框的总数. 最后的维度包含
                `[one-hot 编码的类别标签(独热编码), 人工标准的边界框的 4 个坐标的偏置(相对值), 8 个随意的值(variances)]`
                其中类别标签包括背景类别的标签. 最后的8个值它们的存在只是为了
                使得 `y_true` 和 `y_pred` 的形状一样.最后维度里面的最后4个值为 Anchor 的坐标, 
                在预测的时候需要使用. 如果希望将某个边界框不计入损失函数的计算, 需要将#classes对应
                的one-hot编码的值都设为0.
            y_pred (Keras tensor): 模型预测的输出. 形状和 `y_true` 一样 `(batch_size, #boxes, #classes + 12)`.
                最后一个维度包含如下格式的值
                `[classes one-hot encoded(预测的概率值), 4 predicted box coordinate offsets, 8 arbitrary entries]`.
        Returns:
            张量列表，列表中元素包含批处理batchsize中每张图片的分类损失和定位损失之和。
        '''

        # 转化为Tensor数据类型
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        # 本批次处理的照片数量
        batch_size = tf.shape(y_pred)[0] # 输出类型: tf.int32
        # 本批次中预测到的所有的anchorbox边界框
        n_boxes = tf.shape(y_pred)[1] # 输出类型: tf.int32, `n_boxes` 表达每个图像对应的所有边界框的个数, 不是特征图的每个位置对应的边界框的数量

        # 1: 为每一个边界框计算分类和定位loss：为每张图片的每个anchor box边界框（不分正负样本）计算分类和定位loss
        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # 输出形状: (batch_size, n_boxes)   0:-12  ：6维的独热编码
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # 输出形状: (batch_size, n_boxes)   -12:-8  ：4维度相对坐标值

        # 2: 计算正负样本的分类 losses 

        # 创建正样本、负样本的 mask.
        negatives = y_true[:,:,0] # 形状 (batch_size, n_boxes)  0维是背景
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # 形状 (batch_size, n_boxes)  1， 2， 3， 4， 5维度找出除背景以外的物体类别

        # 计算 y_true 中整个 batch 中正样本 (类别标签 1 到 n) 的数量.
        n_positive = tf.reduce_sum(positives)

        # 不考虑负样本的情况, 计算每一幅图图的正样本的 losses
        # (Keras 计算 loss 的时候只计算每一幅图的损失, 而不是整个 batch 的损失之和, 所以我们需要求和).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # 得到每张图片正样本的损失  Tensor of shape (batch_size,)

        # 计算负样本的分类 loss.

        # 计算所有负样本anchorbox边界框的 loss.
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        # 计算所有batch_size张图片中负样本anchorbox边界框的数量
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) #tf.count_nonzero：在指定的维度上计算非零元素的个数
        
        # 计算我们需要保留的负样本的数量. 最多保留 `self.neg_pos_ratio` 乘以 `y_true` 中正样本的数量, 但是至少 `self.n_neg_min` 个.
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)    #minimum；；maximum限定负样本的个数

        # 完全没有负样本， 所有负样本的分类 loss 为 0, 返回 0 作为 `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])

        # 如果负样本数量不为0，则计算负样本的loss值
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # 将所有的负样本anchorbox边界框的损失变成一维向量（行向量）方便后续的处理
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            # 选择负样本anchorbox边界框中损失最大的前k个负样本，并返回它们得值和下标索引
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)    #是根据损失的大小来选择，保留损失在前K大的负样本
            # 保留选择出的前k个负样本anchorbox边界框中，丢掉一部分负样本，形成新的负样本mask
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) #scatter_nd中的参数：indices是二维的 update是一维向量 shape是一维向量的shape
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # 形状张量(batch_size, n_boxes)
            # 将所有选择出的负样本保存在新的负样本mask之中，并计算出每张图片中负样本的分类损失之和（每张图片中）
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        #总的分类损失（每张图片中）
        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3: 计算定位损失（只考虑正样本的定位损失，不计算负样本的定位损失）
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: 计算整个的损失（每张图片中）alpha越看重分类的损失，希望分类准确哪怕牺牲一些定位的损失
        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`

        # Keras中有一个烦人的习惯，那就是将损失除以批处理大小。
        # 在我们的例子中，这很糟糕，因为平均损失的相关标准是批处理中的正样本(我们在上面的行中除以它)，而不是批处理大小。
        # 所以为了恢复Keras的平均批大小，我们必须乘以它。
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
