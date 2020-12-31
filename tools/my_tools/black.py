import cv2
import numpy as np
import os
from scipy import stats
import scipy.io as sio
from PIL import Image
import copy


def putpalette(img):
	out_img = Image.fromarray(img.squeeze().astype('uint8'))
	sarvocpallete = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255]
	out_img.putpalette(sarvocpallete)
	out_img.show()

def black_area(im_data1, pre_lab):
	# load  the  corresponding  original    data.    use    two    channel is enough
	# a1 = im_data1  # HH channel
	# a4 = im_data2  # VV channel
	# b = np.where(im_data1 == 0)  # find the index where a1==0 and a4==0

	c = im_data1 != 0
	z = copy.deepcopy(pre_lab)

	z[~c] = 0

	# b0 = b[0].shape
	# c = np.ones(im_data1.shape).astype("uint8")
	#
	#
	#
	# for i in range(0, b0[0]):
	# 	c[b[0][i]][b[1][i]] = 0  # 找出原图中黑色区域，做成mask c，c中0元素对应的为黑色，其它为1
	# 	z[b[0][i]][b[1][i]] = 0  # z为最终要输出的图，先将确定的黑色区域变成0

	d = np.where(pre_lab == 0)  # 找出预测图为0的位置
	d0 = d[0].shape

	for i in range(0, d0[0]):
		if (c[d[0][i]][d[1][i]]) == 1:
			q = []
			hw = 40
			while q == []:
				f0 = np.lib.pad(pre_lab, ((hw, hw), (hw, hw)), 'constant')  # 将图像做对称扩展
				p = f0[d[0][i]:d[0][i] + 2 * hw - 1, d[1][i]:d[1][i] + 2 * hw - 1]  # 取出该点对应的patch
				e = np.nonzero(p)  # 将其label中 非零元素的众数作为该点的label
				e0 = e[0].shape
				for j in range(0, e0[0]):
					q.append(p[e[0][j]][e[1][j]])
				if q == []:
					hw += 40
			z[d[0][i]][d[1][i]] = stats.mode(q)[0][0]
	putpalette(z)
	return z

	return z


if __name__ == "__main__":
	pre_lab = cv2.imread("/home/sun/contest/PyTorch-Encoding/outdir/325_feature.png", 0)
	im_data1 = cv2.imread("/home/sun/contest/data_preprocessed/HH/325_HH.tiff", -1)
	# im_data2 = cv2.imread("./blacktest/28_HV.tiff", -1)
	## 黑色区域mask
	if 0 in pre_lab:  # 判断矩阵是否有0元素# there exists black--0 class
		pre_labnew = black_area(im_data1, pre_lab)
