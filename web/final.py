import numpy as np
import os
import random
import glob

import matplotlib.cm as cm

from math import floor, ceil, pow, sqrt

from skimage import img_as_bool, img_as_uint
from skimage import data, io, filters
import skimage.morphology as morphology

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.measure import label, regionprops
from skimage.color import label2rgb

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate, rescale

from skimage.util import pad

from skimage.morphology import skeletonize





MIN_ROTATION = 2
MIN_OBJECT_SIZE = 6
MIN_LETTER_SIZE = 10

NORMALIZED_HEIGHT = 20
NORMALIZED_WIDTH = 10






def get_rotation_angle(image):

    image = morphology.binary_dilation(~image, morphology.rectangle(5,20))

    edges = image

    hough_lines = probabilistic_hough_line(edges)

    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) else 0 for (x1,y1), (x2, y2) in hough_lines]

    rad_angles = [np.arctan(x) for x in slopes]

    deg_angles = [np.degrees(x) for x in rad_angles]

    histo = np.histogram(deg_angles, bins=180)
    
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90-rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number


def rotate_image(image):
    rotation_angle = get_rotation_angle(image)

    image = rotate(image, rotation_angle, resize = True, cval = np.amax(image))

    return img_as_bool(image)


def denoise(image):
    denoised = morphology.remove_small_objects(~image, MIN_OBJECT_SIZE)
    final = denoised
    return ~final


def estimate_letter_dimensions(image):
    
    label_image = label(~image, background=0)

    get_dimensions = lambda bbox: (bbox[3] - bbox[1], bbox[2] - bbox[0])

    label_areas = sorted([get_dimensions(region.bbox) for region in regionprops(label_image) if region.area >= MIN_LETTER_SIZE])
    mid = len(label_areas) // 2

    return label_areas[mid]


def get_paragraphs(image, letter_dimensions):
    lwidth, lheight = letter_dimensions
    larea = lheight * lwidth
    for k in np.linspace(0.5, 2.0, num=4):
        dilated = morphology.binary_dilation(~image, morphology.disk(lheight*k))
        
        label_image = label(dilated, background=0)
        ## key = lambda x : x[0] // x[0] = min.rows (y-it vsortavt)
        paragraph_bboxes = sorted([region.bbox for region in regionprops(label_image) if region.area >= larea], key = lambda x : x[0])
       
        check_paragraph_spacing = True
        for i in range(1,len(paragraph_bboxes)):
            prev = paragraph_bboxes[i-1][2]
            curr = paragraph_bboxes[i][0]
            check_paragraph_spacing = check_paragraph_spacing and (curr - prev) >= lheight
           
        if check_paragraph_spacing:
            return paragraph_bboxes
   
    return paragraph_bboxes



def get_lines(image, letter_dimensions, paragraph_dimensions):
    lwidth, lheight = letter_dimensions
    minr, minc, maxr, maxc = paragraph_dimensions
    paragraph = image[minr:maxr, minc:maxc]

    dilated = morphology.binary_dilation(~paragraph, morphology.rectangle(1, 3*lwidth))

    label_image = label(dilated, background=0)

    line_bboxes = sorted([region.bbox for region in regionprops(label_image) if region.area >= MIN_LETTER_SIZE], key = lambda x : x[0])

    ## paragraphis koordinatebidan imageis coordinatebshi gadayvana
    line_bboxes = [(x[0] + minr, x[1] + minc, x[2] + minr, x[3] + minc) for x in line_bboxes]

    return line_bboxes


def get_words(image, letter_dimensions, line_dimensions):
    lwidth, lheight = letter_dimensions
 
    minr, minc, maxr, maxc = line_dimensions
    line = image[minr:maxr, minc:maxc]
 
    lineWidth = maxc-minc
    lineHeight = maxr-minr
 
    column_sum = np.min(line, axis=0)
 
    left = 0
    while column_sum[left] == 1:
        left += 1
 
    right = lineWidth - 1
    while column_sum[right] == 1:
        right -= 1
 
    intervals = []
 
    i = left
    while i <= right:
        if column_sum[i] == 1:
            intervalLeft = i
            while column_sum[i] == 1:
                i += 1
            intervalRight = i - 1
 
            intervals.append( (intervalLeft,intervalRight) )
 
        else:
            i += 1
 
 
    length = lambda x: x[1]-x[0]
 
    intervals = [interval for interval in intervals if length(interval) > lwidth * 0.5]
 
    space_delims = [left]
    for interval in intervals:
        space_delims.append(interval[0]-1)
        space_delims.append(interval[1]+1)
    space_delims.append(right)
 
 
    word_bboxes = []
 
    for i in range(0,len(space_delims),2):
        curr_minc = space_delims[i]
        curr_maxc = space_delims[i+1]
        word_bboxes.append( (0,curr_minc,lineHeight-1,curr_maxc) )
 
    word_bboxes = [(x[0] + minr, x[1] + minc, x[2] + minr, x[3] + minc) for x in word_bboxes]
 
    return word_bboxes


def get_characters(image, word_dimensions, letter_dimensions):
    minr, minc, maxr, maxc = word_dimensions
    word = image[minr:maxr, minc:maxc]

    # plt.figure()
    # plt.imshow(word, cmap='gray')

    label_image = label(~word, background = 0)

    # plt.figure()
    # plt.imshow(label_image, cmap='gray')

    # plt.show()


    char_bboxes = sorted([region.bbox for region in regionprops(label_image) if region.area >= MIN_LETTER_SIZE], key = lambda x : x[1])
    char_bboxes = [(x[0] + minr, x[1] + minc, x[2] + minr, x[3] + minc) for x in char_bboxes]

    return char_bboxes



def ensure_dir(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)


def generate_training_data_from_file(image, characters, text_file, main_addr):
    ensure_dir(main_addr)
    char_index = 0
    while 1:
        if char_index >= len(characters): break
        char = text_file.read(1)          # read by character
        if not char: break
        if(char == ' ' or char == '\n'): continue
        minr, minc, maxr, maxc = characters[char_index]
        if((maxr-minr)*(maxc-minc) > MIN_OBJECT_SIZE): 
            sub_addr = main_addr+'/'+char
           
            ensure_dir(sub_addr)
            next_num = len(glob.glob(sub_addr+'/*'))

            normalized_image = normalize_character_image(image[minr:maxr, minc:maxc])
         
            io.imsave(sub_addr + '/' + char + str(next_num) + '.bmp', normalized_image*255, cmap = cm.gray)

        char_index += 1



def giglem_gvihvele(char, classifier):
    # next_num = len(glob.glob("../asoebi"+'/*'))
    # io.imsave("../asoebi/" + str(next_num) + '.bmp', char*255, cmap = cm.gray)
    return classifier.classify_image(np.reshape(char, (1, NORMALIZED_HEIGHT, NORMALIZED_WIDTH)))
    # return 'ა'


def recognize_characters(image, characters, output_text_file, classifier):
    for ch in characters:
        if ch in [' ', '\n', '\n\n']:
            output_text_file.write(ch)
            continue
        minr, minc, maxr, maxc = ch
        if (maxr-minr)*(maxc-minc) > MIN_OBJECT_SIZE: 
            normalized_image = normalize_character_image(image[minr:maxr, minc:maxc])
            output_text_file.write(giglem_gvihvele(normalized_image, classifier))


def save_images(image, words, main_addr):
    ensure_dir(main_addr)
    for word in words:
        minr, minc, maxr, maxc = word
        next_num = len(glob.glob(main_addr+'/*'))
        io.imsave(main_addr + '/' +  str(next_num) + '.bmp', image[minr:maxr, minc:maxc]*255, cmap = cm.gray)



def normalize_character_image(char_image):
    height, width = char_image.shape
    height_ratio = NORMALIZED_HEIGHT/height
    width_ratio = NORMALIZED_WIDTH/width
    scale_factor = min(height_ratio, width_ratio)

    char_image = rescale(char_image, scale_factor)
    char_image = img_as_bool(char_image)

    newheight, newwidth = char_image.shape

    width_padding = NORMALIZED_WIDTH - char_image.shape[1]
    height_padding = NORMALIZED_HEIGHT- char_image.shape[0]

    width_padding_tuple = (floor(width_padding/2), ceil(width_padding/2))
    height_padding_tuple =  (floor(height_padding/2), ceil(height_padding/2))

    return pad(char_image, [height_padding_tuple, width_padding_tuple], mode = 'constant', constant_values = 1) 





def split_at_weakest(image, char):
    max_col = 0
    max_col_indx = -1
    minr, minc, maxr, maxc = char
    char_box = image[minr:maxr, minc:maxc]

    for i in range(len(char_box[0])):
        tmp_col = 0
        for j in range(len(char_box)):
            tmp_col += char_box[j,i]
        if tmp_col > max_col:
            max_col_indx = i  
            max_col = tmp_col  

    if sorted(char_box[:,max_col_indx])[0] + sorted(char_box[:,max_col_indx])[1] < 100:
        return None

    if max_col_indx != -1:
        return (minr, minc, maxr, minc+max_col_indx), (minr, minc+max_col_indx+1, maxr, maxc)
    return None




def check_characters(image, characters):
    mean_weight, mean_deviation = get_mean_and_deviation(image, characters)
    # print(mean_weight, mean_deviation, "mean/deviation")
    for i in range (len(characters)):
        minr, minc, maxr, maxc = characters[i]
        char_box = image[minr:maxr, minc:maxc]
        char_box_sum = get_sum(char_box)
        if char_box_sum > mean_weight + 1.5*mean_deviation:
            
            new_boxes = split_at_weakest(image, characters[i])
            if new_boxes != None:
                newbox1, newbox2 = new_boxes
                minr, minc, maxr, maxc = newbox1
                char_box = image[minr:maxr, minc:maxc]

                minr, minc, maxr, maxc = newbox2
                char_box = image[minr:maxr, minc:maxc]

                characters.pop(i)
                characters.insert(i, newbox1)
                characters.insert(i+1, newbox2)
            


def get_sum(arr):
    return sum(map(sum,arr))


def get_mean_and_deviation(image, characters):
    mean_weight = get_mean_character_weight(image, characters)
    mean_deviation = 0
    for ch in characters:
        minr, minc, maxr, maxc = ch
        char_box = image[minr:maxr, minc:maxc]
        mean_deviation += pow((get_sum(char_box) - mean_weight), 2)
    return (mean_weight, sqrt(mean_deviation/(len(characters))))


def get_mean_character_weight(image, characters):
    allpixels = 0
    for ch in characters:
        minr, minc, maxr, maxc = ch
        char_box = image[minr:maxr, minc:maxc]
        allpixels += get_sum(char_box)
    return allpixels/len(characters)



def image_preprocessing(img):

    threshold = filters.threshold_otsu(img)
    img = img > threshold

    img = rotate_image(img)

    img = denoise(img)

    return img



def segment_characters(img_filename):
    img = io.imread(img_filename, as_gray = True)
    # img_original = img.copy()
    img = image_preprocessing(img)


    letter_dimensions = estimate_letter_dimensions(img)

    paragraph_boxes = get_paragraphs(img, letter_dimensions)

    all_chars = []
    all_words = []

    for p in paragraph_boxes:
        lines = get_lines(img, letter_dimensions, p)
        for l in lines:
            words = get_words(img, letter_dimensions, l)
            all_words.extend(words)
            for w in words:
                characters = get_characters(img, w, letter_dimensions)
                all_chars.extend(characters)
                all_chars.append(' ')
            all_chars.append('\n')
        all_chars.append('\n\n')

    # check_characters(img_original, all_chars)
    return img, all_chars



def generate_data_from_file(img_filename, txt_filename, final_folder_name):
    img, all_chars = segment_characters(img_filename)
    all_chars = [x for x in all_chars if x not in [' ', '\n', '\n\n']]
     #es text file aris sadac weria gadmocemuli suratis shesabamisi texti trainingistvis.
    text_file =  open(txt_filename,"r")
    generate_training_data_from_file(img, all_chars, text_file, os.getcwd() + '/' + final_folder_name)

    text_file.close()



def recognize_text_from_image(img_filename, output_txt_filename, classifier):
    img, all_chars = segment_characters(img_filename)
    text_file =  open(output_txt_filename,"w+")
    recognize_characters(img, all_chars, text_file, classifier)
    text_file.close()


def main():
    # for i in range (15, 10, -1):
    #     img_filename = '../tiff/sc'+str(i)+'.tiff'
    #     txt_filename = '../meri.txt'
    #     generate_data_from_file(img_filename, txt_filename, 'meri' + str(i))    

    # img_filename = '../tiff/sc'+str(1)+'.tiff'
    # txt_filename = '../meri.txt'
    # generate_data_from_file(img_filename, txt_filename, 'meri' + str(1))  

    img_filename = '../test data/mesaflave2.tiff'
    txt_filename = '../test data/mesaflave.txt'
    generate_data_from_file(img_filename, txt_filename, 'asdsdsadas' + str(1))    

    # recognize_text_from_image(img_filename, txt_filename)


def main2():
    # img_filename = 'cutlet/sylfaen/text_00.bmp'
    # txt_filename = 'cutlet/sylfaen/text_00.txt'

    # img_filename = 'img/testdata/random4_glaho.tiff'
    # txt_filename = 'random_texts/rand4.txt'

    img_filename = 'random1_arial.tiff'
    txt_filename = 'rand1.txt'

    # img_filename = 'img/testdata/random4_sylfaen.tiff'
    # txt_filename = 'random_texts/rand4.txt'

    generate_data_from_file(img_filename, txt_filename, txt_filename + str(random.randrange(90)))

    

if __name__ == '__main__':
    main()



























