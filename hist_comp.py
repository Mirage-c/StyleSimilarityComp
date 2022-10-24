import cv2
import numpy as np
import os
from tqdm import *
from argparse import ArgumentParser

def get_rgb_hist(img, bsize = 16):
    B = img[..., 0] // ( 256 // bsize )
    G = img[..., 1] // ( 256 // bsize )
    R = img[..., 2] // ( 256 // bsize )
    index = B * bsize * bsize + G * bsize + R
    # 该处形成的矩阵即为直方图矩阵
    rgbhist = cv2.calcHist([index], [0], None, [bsize * bsize * bsize], [0, bsize * bsize * bsize])
    return rgbhist

def getHsvHist(imgHSV):
    '''
    opencv hsv 范围:
    h(0,180)
    s(0,255)
    v(0,255)
    '''
    
    height, width, _ = imgHSV.shape
    H = np.zeros((height, width), dtype=np.uint8)
    S = np.zeros((height, width), dtype=np.uint8)
    V = np.zeros((height, width), dtype=np.uint8)

    h = imgHSV[..., 0]
    s = imgHSV[..., 1]
    v = imgHSV[..., 2]

    h = 2*h
    H[(h > 315) | (h <= 20)] = 0
    H[(h > 20) & (h <= 40)] = 1
    H[(h > 40) & (h <= 75)] = 2
    H[(h > 75) & (h <= 155)] = 3
    H[(h > 155) & (h <= 190)] = 4
    H[(h > 190) & (h <= 270)] = 5
    H[(h > 270) & (h <= 295)] = 6
    H[(h > 295) & (h <= 315)] = 7

    '''
    255*0.2 = 51
    255*0.7 = 178
    '''
    S[s <= 51] = 0
    S[(s > 51) & (s <= 178)] = 1
    S[s > 178] = 2

    V[v <= 51] = 0
    V[(v > 51) & (v <= 178)] = 1
    V[v > 178] = 2

    g = 9*H + 3*S + V
    hist = cv2.calcHist([g], [0], None, [72], [0, 71])
    return hist




def likelihood(img1, img2):
    hist1 = getHsvHist(img1)
    hist2 = getHsvHist(img2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def getImgsFromPath(dir, images):
    assert os.path.isdir(dir) or os.path.islink(
            dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ref', help='reference picture folder', type=str)
    parser.add_argument('--gen', help='generated picture folder', type=str)
    parser.add_argument(
        '--worefgen', default=None, type=str)
    parser.add_argument('--rgb', help='use rgb histogram', default=True)
    args = parser.parse_args()
    # get mapping relationship of filenames between ref and gen
    f = open('./ref_cos.txt', 'r')
    ref_dict = {}
    for line in f:
        a = line.strip().split(',')
        ref_dict[a[0]] = a[1]
    # get imgs from ref and gen folder
    ref_imgs = []
    getImgsFromPath(args.ref, ref_imgs)
    gen_imgs = []
    getImgsFromPath(args.gen, gen_imgs) # 命名与label一致
    # compare them
    tot_score = 0
    if args.rgb:
        for gen_img_path in tqdm(gen_imgs):
            gen_img = cv2.imread(gen_img_path)
            label_file = gen_img_path.split("/")[-1]
            img_file = ref_dict[label_file].replace(".png",".jpg")
            ref_img_path = "/".join([args.ref, img_file])
            ref_img = cv2.imread(ref_img_path)
            
            ref_hist = get_rgb_hist(ref_img)
            gen_hist = get_rgb_hist(gen_img)
            score = cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_CORREL)
            tot_score += score
            continue
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                # ref_hist = cv2.calcHist([ref_img], [i], None, [256], [0,256])
                # gen_hist = cv2.calcHist([gen_img], [i], None, [256], [0,256])
                ref_hist = get_rgb_hist(ref_img)
                gen_hist = get_rgb_hist(gen_img)
                score = cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_CORREL)
                tot_score += score
            # print(f'(1) {gen_img_path} \n (2) {ref_img_path} \n score=', score)
            # exit(0)
    else: # hsv
        for gen_img_path in tqdm(gen_imgs):
            gen_img = cv2.imread(gen_img_path)
            gen_hsv = cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV)
            label_file = gen_img_path.split("/")[-1]
            img_file = ref_dict[label_file].replace(".png",".jpg")
            ref_img_path = "/".join([args.ref, img_file])
            ref_img = cv2.imread(ref_img_path)
            ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
            # wRef_img = cv2.imread("wRef.png")
            # wRef_hsv = cv2.cvtColor(wRef_img, cv2.COLOR_BGR2HSV)
            # woRef_img = cv2.imread("woRef.png")
            # woRef_img = cv2.cvtColor(woRef_img, cv2.COLOR_BGR2HSV)
            score = likelihood(ref_hsv, gen_hsv)
            tot_score += score
            # print(f'(1) {gen_img_path} \n (2) {ref_img_path} \n score=', score)
            # exit(0)
    print('avg_score:', tot_score / len(gen_imgs))