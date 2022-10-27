import cv2
import numpy as np
import os, functools
import tensorflow as tf
from tqdm import *
from argparse import ArgumentParser
import vgg


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
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
    # test()
    parser = ArgumentParser()
    parser.add_argument('--ref', help='reference picture folder', type=str)
    parser.add_argument('--gen', help='generated picture folder', type=str)
    parser.add_argument('--vgg_path', help='path to vgg-19', type=str)
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
    tot_score = np.zeros(5)
    index = 0

    style_target = cv2.imread(gen_imgs[0])

    style_shape = (1,) + style_target.shape
    style_features = {}
    gen_features = {}
    _style_loss = 0
    # precompute style features
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(args.vgg_path, style_image_pre)
        for gen_img_path in tqdm(gen_imgs):
            style_losses = []
            gen_target = cv2.imread(gen_img_path)
            label_file = gen_img_path.split("/")[-1]
            img_file = ref_dict[label_file].replace(".png",".jpg")
            ref_img_path = "/".join([args.ref, img_file])
            style_target = cv2.imread(ref_img_path)
            style_pre = np.array([style_target])
            gen_pre = np.array([gen_target])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={style_image:style_pre})
                features2 = net[layer].eval(feed_dict={style_image:gen_pre})
                # 此后均为具体数值计算
                features = np.reshape(features, (-1, features.shape[3]))
                features2 = np.reshape(features2, (-1, features2.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                gram2 = np.matmul(features2.T, features2) / features2.size
                style_features[layer] = gram
                gen_features[layer] = gram2
                # style_losses.append(2 * tf.nn.l2_loss(gram - gram2)/gram.size)
                style_losses.append(np.sum(np.square(gram - gram2))/gram.size)
            score = style_losses
            tot_score += score 
            score_out = ["{:.1f}".format(x) for x in score]
            tot_score_out = ["{:.1f}".format(x) for x in tot_score]
            print(f"[{index}] {score_out}, tot_score: {tot_score_out}")    
            index += 1
    print('avg_score:', tot_score / len(gen_imgs))