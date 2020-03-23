
'''
    测试模型
'''
from Preject4_目标检测.keras_loss_function.keras_ssd_loss import SSDLoss
from keras import backend as K
from keras.models import load_model
from Preject4_目标检测.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
import matplotlib.pyplot as plt
from Preject4_目标检测.ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
def mask(y_pred_decoded, image):
    line_img = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)
    color= (0,255,0)
    thickness=2
    y_pred_decoded = np.array(y_pred_decoded)[0]
   # y_pred_decoded = np.reshape(y_pred_decoded, (y_pred_decoded[-2], y_pred_decoded[-1]))
    for box in y_pred_decoded:
        xmin = int(box[-4])
        ymin = int(box[-3])
        xmax = int(box[-2])
        ymax = int(box[-1])
        cv2.line(line_img, (xmin, ymin), (xmax, ymin), color=color, thickness=thickness)
        cv2.line(line_img, (xmin, ymin), (xmin, ymax), color=color, thickness=thickness)
        cv2.line(line_img, (xmax, ymin), (xmax, ymax), color=color, thickness=thickness)
        cv2.line(line_img, (xmin, ymax), (xmax, ymax), color=color, thickness=thickness)
        plt.imshow(line_img)
    line_img = np.expand_dims(line_img, axis=0)
#    plt.imshow(line_img)
    return line_img

def Video(image):

    img_height = 300 # 图像的高度
    img_width = 480 # 图像的宽度
    n_classes = 5 # 正样本的类别:汽车、卡车、行人、自行车、交通灯(不包括背景)
    normalize_coords = True # 是否使用相对于图像尺寸的相对坐标

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    K.clear_session() # 从内存中清理曾经加载的模型.
    model_path = 'save_TrainmMode/ssd7_1/ssd7_epoch-25_loss-2.0912_val_loss-1.4135.h5'

    # 加载一个已经训练过的模型,并自定义AnchorBoxes, loss
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                              'compute_loss': ssd_loss.compute_loss})

    batch = len(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image, batch_size=batch)

    i = 0

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)
    a = 0.8
    b = 0.2
    mask_image = mask(y_pred_decoded, image)
    image = cv2.addWeighted(image, a, mask_image, b, 0)
    image = image[0]

    return image


white_output = 'driving_datasets\white.mp4'   #输出文件
clip1 = VideoFileClip("driving_datasets\dfy_driving_data.mp4").resize(newsize=(480, 300, 3))

white_clip = clip1.fl_image(Video) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
clip1.close()