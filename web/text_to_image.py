from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from random import randrange

QUALITY_COEF = 2

FONT_SIZE = 11 * QUALITY_COEF
IN_TEXT_FILE = 'sample_text.txt'
OUT_IMAGE_FILE = 'sample_image.bmp'
OUT_TEXT_FILE = 'sample_text_formatted.txt'

(PAGE_WIDTH, PAGE_HEIGHT) = (595 * QUALITY_COEF, 842 * QUALITY_COEF)
LEFT_MARGIN = 96 * QUALITY_COEF
TOP_MARGIN = 96 * QUALITY_COEF
LINE_SPACING = 6 * QUALITY_COEF

def split_line(text_line, font):
	text_len = len(text_line)
	lines = []

	# LEFT_MARGIN = RIGHT_MARGIN
	max_width = PAGE_WIDTH - 2 * LEFT_MARGIN

	curr_start = 0
	last_word_end = 0

	curr_width = 0
	width_till_last_word = 0

	for i in range(1,text_len):
		if text_line[i] == ' ':
			last_word_end = i
			width_till_last_word = curr_width
		
		curr_width += font.getsize(text_line[i])[0]

		if curr_width > max_width:
			lines.append(text_line[curr_start:last_word_end])
			curr_start = last_word_end + 1
			curr_width = curr_width - width_till_last_word
		elif (i == text_len-1):
			lines.append(text_line[curr_start:])

	return lines




def convert_text_to_image(in_text_filename, image_filename, font_file):
	# create image file
	img = Image.new('L', (PAGE_WIDTH, PAGE_HEIGHT), color = 'white')

	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype(font_file, FONT_SIZE)
	dim_color = randrange(30,50)

	in_txt_file = open(in_text_filename, "r+")

	line_cnt = 0
	paragraph_cnt = 0

	while True:
		text_line = in_txt_file.readline()
		if not text_line: break
		text_line = text_line.strip()

		if len(text_line) == 0:
			paragraph_cnt += 1
			continue

		lines = split_line(text_line, font)

		for line in lines:
			# draw.text((x, y),"Sample Text",(r,g,b))
			y = TOP_MARGIN + (line_cnt + paragraph_cnt) * (FONT_SIZE + LINE_SPACING)
			draw.text((LEFT_MARGIN,y), line, font=font, fill=dim_color)
			line_cnt += 1

	img.save(image_filename, "BMP")

	in_txt_file.close()
	img.close()


if __name__ == '__main__':
	main()