from PIL import Image
import numpy as np
import cv2

# image_path = "/home/sun/contest/data_preprocessed/output_path/500_feature.png"
image_path = "/home/sun/contest/data_preprocessed/output_path/17_gt.png"
# image_path = "/home/sun/contest/data_preprocessed/input_path/17_HH.tif"


def putpalette(img):
	# image = Image.open(image_path)
	# img = numpy.array(image)
	out_img = Image.fromarray(img.squeeze().astype('uint8'))
	sarvocpallete = [0,0,0, 0,255,255, 0,0,255, 255,255,0, 0,255,0, 255,0,0, 255,255,255]
	out_img.putpalette(sarvocpallete)
	out_img.show()

if True:
	# 1.读取图片
	image = Image.open(image_path)
	img = cv2.imread(image_path, -1)
	img = img.astype('int64')
	putpalette(img)



	# 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
	kernel = np.ones((18, 18), np.uint8)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	putpalette(closing)
	# 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
	# kernel = np.ones((18, 18), np.uint8)
	# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	# putpalette(opening)