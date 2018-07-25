import random

import numpy as np

from add_noise import add_noise
from text_to_image import convert_text_to_image
from final import generate_data_from_file

ANBANI = ['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ']
ANBANIENG = list('abgdevzTiklmnopJrstufqRySCcZwWxjh')

SYMBOLS = ANBANI + [' '] * 6
SYMBOLSENG = ANBANIENG + [' '] * 6 



TEXT_LENGTH = 1200
MAX_WORD_LENGTH = 12

def generate_random_text(filename, alphabet, symbols):
	text = [random.choice(symbols) for _ in range(TEXT_LENGTH)]

	# no word should be too long (not to fit on a line)
	curr_word_length = 0
	for i in range(TEXT_LENGTH):
		if curr_word_length == 0 and text[i] == ' ':
			text[i] = random.choice(alphabet)

		if text[i] != ' ':
			curr_word_length += 1
		else:
			curr_word_length = 0

		if curr_word_length == MAX_WORD_LENGTH:
			text[i] = ' '
			curr_word_length = 0

	text = ''.join(text)

	with open(filename, 'w+') as f:
		print(text, file=f)

def generate_random_text_with_spaces(filename, alphabet, symbols):
	text = [random.choice(symbols) for _ in range(TEXT_LENGTH)]

	# no word should be too long (not to fit on a line)
	text = ' '.join(text)

	with open(filename, 'w+') as f:
		print(text, file=f)


SYLFAEN = 'fonts/sylfaen.ttf'
NOTO = 'fonts/noto.ttf'
DEJAVU = 'fonts/dejavu.ttf'
GLAHO = 'fonts/glaho.ttf'
ARIAL =  'fonts/arial.ttf'

font_to_str = {SYLFAEN : 'sylfaen', NOTO : 'noto', DEJAVU : 'dejavu', GLAHO : 'glaho', ARIAL : 'arial'}




def main():
	for i in range(1,11):
		generate_random_text_with_spaces('random%0.2d.txt' % i, ANBANI, ANBANI)



def main3():
	alphabet = ANBANIENG
	symbols =  ANBANIENG
	font_file = LITNUSX

	for i in range(1):
		txt_filename = 'cutlet/%s/text_%0.2d.txt' % (font_to_str[font_file], i)
		img_filename = 'cutlet/%s/text_%0.2d.bmp' % (font_to_str[font_file], i)
		
		folder_name = '_'.join(['asoebi',font_to_str[font_file],('%0.2d' % i)])

		generate_random_text(txt_filename, alphabet, symbols)
		convert_text_to_image(txt_filename, img_filename, font_file)
		add_noise(img_filename)


def main2():
	for font_file in [SYLFAEN, LITNUSX, ACADNUSX, GLAHO]:
		if font_file == SYLFAEN or font_file == GLAHO:
			alphabet = ANBANI
			symbols = SYMBOLS
		else:
			alphabet = ANBANIENG
			symbols = SYMBOLSENG


		for i in range(1):
			txt_filename = 'cutlet/%s/text_%0.2d.txt' % (font_to_str[font_file], i)
			img_filename = 'cutlet/%s/text_%0.2d.bmp' % (font_to_str[font_file], i)
			
			folder_name = '_'.join(['asoebi',font_to_str[font_file],('%0.2d' % i)])

			generate_random_text(txt_filename, alphabet, symbols)
			convert_text_to_image(txt_filename, img_filename, font_file)
			add_noise(img_filename)


		# generate_data_from_file(img_filename, txt_filename, folder_name)


if __name__ == '__main__':
	main()





