import os
from math import floor, ceil

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from skimage import io, transform, util, filters, measure
from skimage import img_as_uint

import random

MAX_ANGLE = 1.0
NORMALIZED_HEIGHT = 20
NORMALIZED_WIDTH  = 10

def draw(character, font, font_size, image_filename):
	font = ImageFont.truetype(font, font_size)
	dim_color = random.randrange(50,80)

	character_size = font.getsize(character)

	img = Image.new('L', character_size, color='white')
	draw = ImageDraw.Draw(img)

	draw.text((0,0), character, font=font, fill=dim_color)

	img.save(image_filename, "BMP")
	img.close()

def add_noise(filename):
	img = io.imread(filename, as_gray=True)

	# rotate a bit
	rangle = random.uniform(-MAX_ANGLE, MAX_ANGLE)
	img = transform.rotate(img, rangle, cval=1.0)

	# add noise
	img = util.random_noise(img, var=0.001)

	return img

def get_bbox(img):
	label_image = measure.label(img, background=1)

	props = measure.regionprops(label_image)

	if len(props) == 1:
		return props[0].bbox
	else:
		return None

def normalize(char_image, threshold):
	height, width = char_image.shape
	height_ratio = NORMALIZED_HEIGHT/height
	width_ratio = NORMALIZED_WIDTH/width
	scale_factor = min(height_ratio, width_ratio)

	# char_image = char_image > threshold
	char_image = transform.rescale(char_image, scale_factor)

	# plt.figure()
	# plt.imshow(char_image, cmap="gray")
	# plt.show()

	char_image = char_image > threshold

	height_padding = NORMALIZED_HEIGHT- char_image.shape[0]
	width_padding = NORMALIZED_WIDTH - char_image.shape[1]
	
	height_padding_tuple =  (floor(height_padding/2), ceil(height_padding/2))
	width_padding_tuple = (floor(width_padding/2), ceil(width_padding/2))
	
	char_image = util.pad(char_image, [height_padding_tuple, width_padding_tuple], \
		mode='constant', constant_values=1)
	return char_image 

def main():
	for c in [chr(ord('ა') + i) for i in range(33)]:
		if not os.path.isdir('nodo/%s' % c):
			os.makedirs('nodo/%s' % c)
		for i in range(100):
			filename = 'nodo/%s/%s%0.2d.bmp' % (c,c,i)
			draw(c, 'fonts/NotoSansGeorgianRegular.ttf', 22, filename)
			img = add_noise(filename)

			img_orig = img.copy()
			thresh = filters.threshold_otsu(img)
			img = img > thresh
			bbox = get_bbox(img)
			if not bbox:
				print('More than one label')
				return
			(minr,minc,maxr,maxc) = bbox
			patch = img_orig[minr:maxr, minc:maxc]

			patch = normalize(patch, thresh)

			io.imsave(filename, img_as_uint(patch))


if __name__ == '__main__':
	main()