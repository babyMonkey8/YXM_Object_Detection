import numpy as np

offset_height = 0.5
offset_width = 0.5
feature_map_height = 37
feature_map_width = 60
img_height = 300
img_width = 480
aspect_ratios = [0.5, 1.0, 2.0]
this_scale = 0.08
next_scale = 0.16
two_boxes_for_ar1 = True

wh_list = []
size = min(img_height, img_width)
for ar in aspect_ratios:
    if (ar == 1):
        # 对应长宽比为 1 的情况计算 anchor .
        box_height = box_width = this_scale * size
        wh_list.append((box_width, box_height))
        if two_boxes_for_ar1:
            # 使用本层和下一层的 scale 的几何平均值计算稍微大一点的 Anchor 的尺寸
            box_height = box_width = np.sqrt(this_scale * next_scale) * size
            wh_list.append((box_width, box_height))
    else:
        box_height = this_scale * size / np.sqrt(ar)
        box_width = this_scale * size * np.sqrt(ar)
        wh_list.append((box_width, box_height))
wh_list = np.array(wh_list)
print(wh_list)
print(wh_list.shape)
print('\n')

step_height = img_height / feature_map_height  # 步长是相对于原始图片横方向上距离的步长
step_width = img_width / feature_map_width  # 步长是相对于原始图片纵方向上距离的步长
cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)  # numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成
cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
print(cy)
print(cx)
print('cx和cy形状：', cx.shape, cy.shape)
print('\n')

cx_grid, cy_grid = np.meshgrid(cx, cy)  #生成网格点坐标矩阵
print(cx_grid)
print(cy_grid)
print('cx_grid和cy_grid：', cx_grid.shape, cy_grid.shape)
print('\n')

cx_grid = np.expand_dims(cx_grid, -1) # 为了如下的 np.tile() 做准备
cy_grid = np.expand_dims(cy_grid, -1) # 为了如下的 np.tile() 做准备
print(cx_grid)
print(cy_grid)
print('cx和cy形状：', cx_grid.shape, cy_grid.shape)
print('\n')

# 产生一个 4 维的 tensor 模版, 形状为 `(feature_map_height, feature_map_width, n_boxes, 4)`
# 最后 4 维的值为 `(cx, cy, w, h)`
n_boxes = 4
boxes_tensor = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))
print(boxes_tensor)
print('boxes_tensor的形状：', boxes_tensor.shape)
print('\n')


boxes_tensor_cx = np.tile(cx_grid, (1, 1, n_boxes))
boxes_tensor_cy = np.tile(cy_grid, (1, 1, n_boxes))
print('boxes_tensor_cx的形状', boxes_tensor_cx.shape)
print('boxes_tensor_cy的形状', boxes_tensor_cy.shape)
print('\n')


boxes_tensor_w = wh_list[:, 0]
boxes_tensor_h = wh_list[:, 1]
print('boxes_tensor_w:', boxes_tensor_w)
print('boxes_tensor_h:', boxes_tensor_w)
print(boxes_tensor_w.shape)
print(boxes_tensor_h.shape)
print('\n')

boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # 设置 cx
boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # 设置 cy
boxes_tensor[:, :, :, 2] = wh_list[:, 0] # 设置 w
boxes_tensor[:, :, :, 3] = wh_list[:, 1] # 设置 h
print(boxes_tensor)
print(boxes_tensor.shape)
print('\n')

x_coords = boxes_tensor[:,:,:,[0, 2]]
xxxxxx = x_coords[x_coords >= img_width]
y_coords = boxes_tensor[:,:,:,[1, 3]]
print(x_coords)
print(x_coords.shape)
print(y_coords)
print(y_coords.shape)
print(xxxxxx)