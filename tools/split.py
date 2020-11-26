"""
-------------
This is the multi-process version
"""
import os
import codecs
import numpy as np
import math
import cv2
import copy
import time
# import gdal
import pdb
import imageio
import xml.etree.ElementTree as ET

# imageio.imwrite()

class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code='utf-8',
                 gap=0,
                 subsize=500,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext='.tiff',
                 padding=True,
                 num_process=8
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'AIR-SARShip-1.0-images')
        self.labelpath = os.path.join(self.basepath, 'AIR-SARShip-1.0-labels')
        self.outimagepath = os.path.join(self.outpath, 'AIR-SARShip-1.0-data')
        self.outlabelpath = os.path.join(self.outpath, 'AIR-SARShip-1.0-xml')
        self.ext = ext
        self.padding = padding
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
    

    def saveimagepatches(self, subimg, subimgname, left, up):
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        subimg = np.repeat(subimg[:, :, np.newaxis], 3, axis=2)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)



    def savepatches(self, subimg, subimgname, left, up,line):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        #if os.path.exists(outdir):
        f_out = open(outdir,'a+')
        #else:
        #    f_out = open(outdir, 'w')
        x0,y0,x1,y1,x2,y2,x3,y3,cls,_ = line.strip().split(' ')
            #outline = ' '.join(list(map(str, [float(x0)-left,float(y0)-up,float(x1)-left,float(y1)-up,float(x2)-left,float(y2)-up,float(x3)-left,float(y3)-up])))
        outline = "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(float(x0)-left,float(y0)-up,float(x1)-left,float(y1)-up,float(x2)-left,float(y2)-up,float(x3)-left,float(y3)-up)
        outline = outline + ' ' + \
            cls + ' ' + str('0')
        f_out.write(outline + '\n')   
        f_out.close()    
        self.saveimagepatches(subimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        print(name)
        img = imageio.imread(os.path.join(self.imagepath, name))
        #pdb.set_trace()
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name[:-4] + 'xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        
        lines = f.readlines()
        f.close()
        outbasename = name.split('.')[0] + '__' + str(rate) + '__'
        weight = np.shape(img)[0]
        height = np.shape(img)[1]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimg = img[left:right,up:down]
                #print(left , right, up,down ,weight,height)
                #print(subimg)
                lmax = np.amax(subimg)
                lmin = np.amin(subimg)
                
                if lmin == lmax:
                    if (up + self.subsize >= height):
                        break
                    else:
                        up = up + self.slide
                    continue

                subimg = (subimg - lmin) / (lmax - lmin + 0.05) * 255.0
                
                for i,line in enumerate(lines):
                    #if i < 2:
                    #    continue
                    x0,y0,x1,y1,x2,y2,x3,y3,cls,_ = line.strip().split(' ')
                    xmax = max(float(x0),float(x1),float(x2),float(x3))
                    xmin = min(float(x0),float(x1),float(x2),float(x3))
                    ymax = max(float(y0),float(y1),float(y2),float(y3))
                    ymin = min(float(y0),float(y1),float(y2),float(y3))
                    if xmin - left >= 0 and right - xmax >= 0 and up - ymin <= 0 and down - ymax >= 0:
                        subimgname = outbasename + str(left) + '___' + str(up)
                        self.savepatches(subimg, subimgname, left, up, line)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        imagelist = os.listdir(self.imagepath)
        for name in imagelist:
            
            self.SplitSingle(name, rate, self.ext)



if __name__ == '__main__':
    split = splitbase(r'/home/sun/projects/sar',r'/home/sun/projects/sar',
                      gap=0,subsize=500,num_process=1)
    split.splitdata(1)
