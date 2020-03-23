'''
    通过本Python脚本来完成VOC数据集格式的生成。
'''
import os,cv2,sys,shutil
from xml.dom.minidom import Document
from tensorflow_tanxinkeji_works.Preject4_目标检测.data_generator.object_detection_2d_data_generator import DataGenerator
import numpy as np

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
eval_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

train_labels_filename = 'driving_datasets\driving_datasets\labels_train.csv'
val_labels_filename = 'driving_datasets\driving_datasets\labels_val.csv'
eval_labels_failename = 'driving_datasets\driving_datasets\labels_eval_dfy.csv'

images_dir = 'driving_datasets\driving_datasets'

train_dataset = train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # CSV 文件前 6 列的值
                        include_classes='all',
                        ret = True)
val_dataset = val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all',
                      ret=True)
eval_dataset = eval_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=eval_labels_failename,
                      input_format=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all',
                      ret=True)


#
# # 得到训练和validation数据集的数据的量.
# train_dataset_size = train_dataset.get_dataset_size()
# val_dataset_size   = val_dataset.get_dataset_size()
# print("训练集的图像数量:\t{:>6}".format(train_dataset_size))
# print("validation集的图像数量\t{:>6}".format(val_dataset_size))



def writexml(filename,saveimg,bboxes,xmlpath,img_set):
    '''
    本函数将用于打包xml格式的文件，是一种标准的打包xml文件打包的方法。
    对于这个函数只需要传入以下几个参数就能够写入一个xml文件到指定的’xmlpath‘文件路径下
    :param filename:图片名
    :param saveimg:图片信息
    :param bboxes:bounding box信息
    :param xmlpath:xml文件存储信息
    :param img_set:数据集类型：训练数据或者测试数据
    :return:
    '''
    doc = Document()

    annotation = doc.createElement('annotation')

    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    if img_set == 'train':
        data_name = 'VOC2007'
    elif img_set =='val':
        data_name = 'VOC2012'
    elif img_set == 'eval':
        data_name = 'VOC2012'
    folder_name = doc.createTextNode(data_name)
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The ' + data_name + ' Database'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL ' + data_name))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        # class_name_mask = ['汽车', '卡车', '行人', '自行车', '交通灯']
        # class_name = str(class_name_mask[bbox[0] - 1])

        class_name = str(bbox[0])

        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(class_name))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[4])))
        bndbox.appendChild(ymax)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()

'''
    我们在生成xml文件时，只需要调用’writexml‘这个函数就能够完成xml文件的写入。
    如何解析wider_face的真值文件，并且生成VOC格式的数据：
        首先我们需要定义根目录：rootdir,这个目录是用来存储VOC格式标注图片和原始图片的路径
        然后我们定义一个函数：’convertimgset‘，用来解析wider_face所对应的真值文件
            convertimgset函数参数’img_set‘:是我们需要解析的文件的路径，包括两个文件：一个train,一个是value
'''
# rootdir = "/media/kuan/新加卷/wider_face"
rootdir = 'driving_datasets/voc/'


def convertimgset(img_set, data):
#    imgdir = rootdir + img_set   # 指向图片文件
    imgdir = data[1] # 图片的全路径
    imgname = data[3] # 图片名
#    gtfilepath = rootdir + img_set + "labels_train.csv"  # 指向标注文件

    fwrite = open(rootdir + "ImageSets/Main/" + img_set + ".txt", 'w')

    index = 0  # 索引，表示我们解析到第几张图片
    num_data = len(imgdir)
 #   with open(gtfilepath, 'r') as gtfiles:  # 打开标注文件
    while index < num_data: #true  # 如果只需要读取n张图片图片可以在此处修改，如只读取1000张图片，可以修改为：while(index < 1000):
        filename = imgdir[index]  #读取其中一行数据，
        single_imgname =imgname[index]
        if filename == None or filename == "":  # 判断该行图片路径是否存在
            break
#        imgpath = imgdir + "/" + filename  # 拼接图片路径

        img = cv2.imread(filename)  # 读取图片，可以拿到图片的数据信息

        if not img.data:  # 判读普片是否存在
            break;


        numbbox = int(len(data[2][index]))

        bboxes = []

        for i in range(numbbox):
            lines = data[2][index][i]
            lines = np.array(lines)
            bbox = (int(lines[0]), int(lines[1] * 0.866), int(lines[2] * 1.3866), int(lines[3] * 0.866), int(lines[4]* 1.3866))

            bboxes.append(bbox)

        if len(bboxes) == 0:
            continue

        # 写入JPEGImages文件(jpg)
        cv2.imwrite("{}/JPEGImages/{}.jpg".format(rootdir,single_imgname), img)

        # 写入ImageSets文件（.txt文件）
        fwrite.write(single_imgname + "\n")

        # 定义xml格式文件名
        xmlpath = "{}/Annotations/{}.xml".format(rootdir, single_imgname)

        #写入Annotations文件(.xml文件）
        writexml(single_imgname, img, bboxes, xmlpath, img_set)

        print("success number is ", index)
        index += 1

    fwrite.close()

if __name__=="__main__":
    # img_sets = ["train","val"]
    # for img_set in img_sets:
    #     if img_set == 'train':
    #         data = train_dataset
    #     else:data = val_dataset
    #     convertimgset(img_set, data)


    # img_sets = 'eval'
    # data = eval_dataset
    # convertimgset(img_sets, data)

    img_sets = 'val'
    data = val_dataset
    convertimgset(img_sets, data)
    # 修改文件名
    # shutil.move(rootdir + "/ImageSets/Main/" + "train.txt", rootdir + "/ImageSets/Main/" + "trainval.txt")
    # shutil.move(rootdir + "/ImageSets/Main/" + "val.txt", rootdir + "/ImageSets/Main/" + "test.txt")
