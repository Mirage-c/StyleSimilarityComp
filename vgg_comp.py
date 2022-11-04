import cv2
import numpy as np
import os, functools
import jittor as jt
from jittor import nn
from jittor import models
from jittor.dataset.dataset import Dataset
from tqdm import *
from argparse import ArgumentParser


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

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19()

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        features = x_vgg[0]
        features2 = y_vgg[0]
        # 此后均为具体数值计算
        features = features.reshape(-1, features.shape[1])
        features2 = features2.reshape(-1, features2.shape[1])
        gram = jt.matmul(features.transpose(), features) / features.numel()
        gram2 = jt.matmul(features2.transpose(), features2) / features2.numel()
        return (jt.sum(jt.pow(gram - gram2, 2))/gram.numel())

class ImageTestSet(Dataset):
    def __init__(self, args, batch_size=1, shuffle=False, num_workers=4):
        super().__init__(num_workers=num_workers)
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get mapping relationship of filenames between ref and gen
        f = open('./ref_cos.txt', 'r')
        self.ref_dict = {}
        for line in f:
            a = line.strip().split(',')
            self.ref_dict[a[0]] = a[1]
        # self.gen_imgs = []
        # self.ref_imgs = []
        self.gen_img_paths = []
        getImgsFromPath(args.gen, self.gen_img_paths) # 命名与label一致
        # for gen_img_path in tqdm(gen_img_paths):
        #     gen_target = cv2.imread(gen_img_path)
        #     label_file = gen_img_path.split("/")[-1]
        #     img_file = ref_dict[label_file].replace(".png",".jpg")
        #     ref_img_path = "/".join([args.ref, img_file])
        #     style_target = cv2.imread(ref_img_path)
        #     style_pre = jt.array(style_target, dtype=jt.float32).transpose(2,0,1)
        #     gen_pre = jt.array(gen_target, dtype=jt.float32).transpose(2,0,1)
        #     self.gen_imgs.append(gen_pre)
        #     self.ref_imgs.append(style_pre)
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.gen_img_paths),
            shuffle=self.shuffle
        )

    def __getitem__(self, idx):
        gen_img_path = self.gen_img_paths[idx]
        gen_target = cv2.imread(gen_img_path)
        label_file = gen_img_path.split("/")[-1]
        img_file = self.ref_dict[label_file].replace(".png",".jpg")
        ref_img_path = "/".join([args.ref, img_file])
        style_target = cv2.imread(ref_img_path)
        style_pre = jt.array(style_target, dtype=jt.float32).transpose(2,0,1)
        gen_pre = jt.array(gen_target, dtype=jt.float32).transpose(2,0,1)
        return style_pre, gen_pre
        # return self.ref_imgs[idx], self.gen_imgs[idx]

if __name__ == "__main__":
    # test()
    parser = ArgumentParser()
    parser.add_argument('--ref', help='reference picture folder', type=str)
    parser.add_argument('--gen', help='generated picture folder', type=str)
    parser.add_argument('--vgg_path', help='path to vgg-19, depreciated here', type=str, default=None)
    args = parser.parse_args()
    if args.vgg_path is not None:
        print("### WARNING: vgg_path is depreciated, and pretrained vgg is used instead ###")
    image_set = ImageTestSet(args, num_workers=0)
    # compare them
    tot_score = jt.float32(0)
    index = jt.int32(0)
    _style_loss = 0
    vgg_loss = VGGLoss()
    vgg_loss.eval()
    for ref_img, gen_img in tqdm(image_set):
        # send them into the vgg net
        score = vgg_loss(ref_img, gen_img)
        tot_score += score
        ### OUTPUT ###
        # score_out = ["{:.1f}".format(x) for x in score]
        # tot_score_out = ["{:.1f}".format(x) for x in tot_score]
        # print(f"[{index}] {score_out}, tot_score: {tot_score_out}")    
        ### index step ###
        # index += 1
        # if index % 200 == 0:
        #     print(tot_score.data[0])
    print('avg_score:', tot_score.data[0] / len(image_set.gen_img_paths))