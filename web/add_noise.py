import random

from skimage import io, transform, filters, util

MAX_ROTATION_ANGLE = 1.0
BLUR_SIGMA = 0.2

def add_noise(in_file, out_file=None):
	img = io.imread(in_file, as_gray=True)

	# rotate by a small angle
	rangle = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
	img = transform.rotate(img, rangle, cval=1)

	# blur a bit
	util.random_noise(img)

	if out_file:
		io.imsave(out_file, img)
	else:
		io.imsave(in_file, img)


def main():
	add_noise('text_0.bmp', 'text_0_noisy.jpg')

if __name__ == '__main__':
	main()
