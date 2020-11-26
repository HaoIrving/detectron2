"""
对原始遥感图像训练数据进行裁切，生成固定大小的patches,适用于HBB(Horizontal Bounding Box)
"""

import cv2
import os
import sys
import numpy as np
import glob
from multiprocessing import Pool
from functools import partial
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
from lxml import etree
import imageio
from libtiff import TIFF

def iou(BBGT, imgRect):
    """
    并不是真正的iou。计算每个BBGT和图像块所在矩形区域的交与BBGT本身的的面积之比，比值范围：0~1
    输入：BBGT：n个标注框，大小为n*4,每个标注框表示为[xmin,ymin,xmax,ymax]，类型为np.array
          imgRect：裁剪的图像块在原图上的位置，表示为[xmin,ymin,xmax,ymax]，类型为np.array
    返回：每个标注框与图像块的iou（并不是真正的iou），返回大小n,类型为np.array
    """
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])
    wh = np.maximum(right_bottom-left_top, 0)
    inter_area = wh[:, 0]*wh[:, 1]
    iou = inter_area/((BBGT[:, 2]-BBGT[:, 0])*(BBGT[:, 3]-BBGT[:, 1]))
    return iou


def get_bbox(xml_path):
    BBGT = []
    # f = open(txt_path, 'r')
    # liness = f.readlines()
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    # print("liness", len(liness))
    # for lines in liness:
    #     lines = lines.split(',')
    #     if len(lines) == 5:
    #         lines = [line.strip() for line in lines]
    #         xmin = int(lines[0][1:])
    #         ymin = int(lines[1][:-1])
    #         xmax = int(lines[2][1:])
    #         ymax = int(lines[3][:-1])
    #         label = int(lines[4])
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        xmin = int(bbox.find('xmin').text) 
        ymin = int(bbox.find('ymin').text) 
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        BBGT.append([xmin, ymin, xmax, ymax, 0])
    return np.array(BBGT)


def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]


def split(imgname, dirsrc, dirdst, class_dict, subsize=800, gap=200, iou_thresh=0.3, ext='.png'):
    """
    imgname:   待裁切图像名（带扩展名）
    dirsrc:    待裁切的图像保存目录的上一个目录，默认图像与标注文件在一个文件夹下，图像在images下，标注在labelTxt下，标注文件格式为每行一个gt,
               格式为xmin,ymin,xmax,ymax,class,想读其他格式自己动手改
    dirdst:    裁切的图像保存目录的上一个目录，目录下有images,labelTxt两个目录分别保存裁切好的图像或者txt文件，
               保存的图像和txt文件名格式为 oriname_min_ymin.png(.txt),(xmin,ymin)为裁切图像在原图上的左上点坐标,txt格式和原文件格式相同
    subsize:   裁切图像的尺寸，默认为正方形，想裁切矩形自己动手改
    gap:       相邻行或列的图像重叠的宽度
    iou_thresh:小于该阈值的BBGT不会保存在对应图像的txt中（在图像过于边缘或与图像无交集）
    ext:       保存图像的格式
    """
    img = cv2.imread(os.path.join(os.path.join(dirsrc,'AIR-SARShip-1.0-images'), imgname), -1)
    xml_path = os.path.join(os.path.join(dirsrc, 'AIR-SARShip-1.0-labels'), imgname[:-5] +'-label.xml')
    BBGT = get_bbox(xml_path)
    print("before crop : %d" % len(BBGT))
    count = 0
    
    img_h,img_w = img.shape[:2]
    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize >= img_h:
            reachbottom = True
            top = max(img_h-subsize, 0)
        while not reachright:
            if left + subsize >= img_w:
                reachright = True
                left = max(img_w - subsize, 0)
            imgsplit = img[top: min(top + subsize, img_h), left: min(left+subsize, img_w)]
            if imgsplit.shape[:2] != (subsize, subsize):
                template = np.zeros((subsize, subsize), dtype=np.uint16)
                template[0: imgsplit.shape[0], 0: imgsplit.shape[1]] = imgsplit
                imgsplit = template
            imgrect = np.array([left, top, left + subsize, top + subsize]).astype('float32')
            ious = iou(BBGT[:, : 4].astype('float32'), imgrect)
            BBpatch = BBGT[ious > iou_thresh]
            ## abandaon images with 0 bboxes
            
            if len(BBpatch) > 0:
                print("ious:")
                print(ious)
                print("after crop: %d" % len(BBpatch))
                print("\n")
                count += len(BBpatch)
                path = os.path.join(os.path.join(dirdst, 'AIR-SARShip-1.0-data'),
                                        imgname[: -5] + '_' + str(left) + '_' + str(top) + ext)
                tif = TIFF.open(path, mode='w')
                tif.write_image(imgsplit)
                # cv2.imwrite(os.path.join(os.path.join(dirdst, 'JPEGImages'),
                #                          imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), imgsplit)
                xml = os.path.join(os.path.join(dirdst, 'AIR-SARShip-1.0-xml'),
                                        imgname[: -5] + '_' + str(left) + '_' + str(top) + '.xml')
                ann = GEN_Annotations(path)
                ann.set_size(imgsplit.shape[0], imgsplit.shape[1])
                for bb in BBpatch:
                    x1, y1, x2, y2, target_id = int(bb[0]) - left, int(bb[1]) - top, int(bb[2]) - left, int(bb[3]) - top, int(bb[4])
                    # target_id, x1, y1, x2, y2 = anno_info
                    label_name = get_key(class_dict, int(target_id))[0]
                    ann.add_pic_attr(label_name, x1, y1, x2, y2)
                ann.savefile(xml)

            left += subsize-gap
        top+=subsize-gap
    print("count %d" % count)


class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = "AIR-SARShip-1.0-data"

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        # child3 = etree.SubElement(self.root, "source")

        # child4 = etree.SubElement(child3, "annotation")
        # child4.text = "PASCAL VOC2007"
        # child5 = etree.SubElement(child3, "database")
        # child5.text = "Unknown"

        # child6 = etree.SubElement(child3, "image")
        # child6.text = "flickr"
        # child7 = etree.SubElement(child3, "flickrid")
        # child7.text = "35435"

    def set_size(self, witdh, height):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        # channeln = etree.SubElement(size, "depth")
        # channeln.text = str(channel)

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)

if __name__ == '__main__':
    import tqdm
    dirsrc= '/home/sun/projects/sar'      #待裁剪图像所在目录的上级目录，图像在images文件夹下，标注文件在labelTxt下
    dirdst= '/home/sun/projects/sar'   #裁剪结果存放目录，格式和原图像目录一样
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'AIR-SARShip-1.0-data')):
        os.mkdir(os.path.join(dirdst, 'AIR-SARShip-1.0-data'))
    if not os.path.exists(os.path.join(dirdst, 'AIR-SARShip-1.0-xml')):
        os.mkdir(os.path.join(dirdst, 'AIR-SARShip-1.0-xml'))

    class_dict = {"ship": 0}
    subsize = 500
    gap = 300
    iou_thresh = 0.8
    ext = '.tiff'

    imglist = glob.glob(f'{dirsrc}/AIR-SARShip-1.0-images/*.tiff')
    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist]
    for imgname in tqdm.tqdm(imgnameList):
        split(imgname, dirsrc, dirdst, class_dict, subsize, gap, iou_thresh, ext)
