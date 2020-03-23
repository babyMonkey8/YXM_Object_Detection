'''
Utilities to match ground truth boxes to anchor boxes.
'''

import numpy as np

def match_bipartite_greedy(weight_matrix):
    '''
    Returns a bipartite matching according to the given weight matrix.
    严格匹配
    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    '''

    weight_matrix = np.copy(weight_matrix) # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes)) # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # 在下面循环的每次迭代中，都有一个人工标注框，每个人工标注框将要匹配到一个anchor box.
    for _ in range(num_ground_truth_boxes):

        # 通过两个步骤找到最大的anchor box-人工标注框对:首先，在anchor box上进行约简，然后在人工标注框上进行约简。
        anchor_indices = np.argmax(weight_matrix, axis=1) # 为每个人工标注框匹配到IOU最大的anchor box的下标
            # 补充：本来np.argmax(weight_matrix, axis=1)这条语句就可以为每个人工标注框匹配到最合适anchor box, 那么为什么还需要循环并执行下面的语句呢？
                # 因为可能存在一种情况即同一个anchor box和多个人工标注框匹配度在这几个人工标注框中都是最高的，但是一个anchor box和一个人工标注框匹配度最佳
                # 那么我们需要在已经np.argmax之后在选择一次。以此形成一个anchor box与一个人工标注框匹配度最高
        overlaps = weight_matrix[all_gt_indices, anchor_indices] # 找出每个人工标注框匹配到IOU最大的anchor box的值
        ground_truth_index = np.argmax(overlaps) # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index # Set the match.

        # 将匹配的人工标注框的行和匹配的anchor box框的列设置为所有零。这将确保不会再次匹配这些框，因为它们永远不会是其他任何框的最佳匹配。
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches

def match_multi(weight_matrix, threshold):
    '''
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        两个长度相等的一维数字数组，表示匹配的索引。第一个数组包含' weight_matrix '的第一个轴上的索引，第二个数组包含沿第二轴的指标。
    '''

    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes)) # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0) # Array of shape (weight_matrix.shape[1],)
    # 补充：在np.argmax(weight_matrix, axis=0)中，为什么使用axis =0方式而不是直接使用axis = 1同时>=阈值的方法
        # 因为人工标注框和anchor box的匹配关系是1对1的、1对多的关系，不能存在多对1的关系
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices] # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]  # numpy.nonzero() 函数返回输入数组中非零元素的索引。
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met
