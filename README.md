# StyleSimilarityComp
尝试比较图片间的风格相似度
## 直方图法
rgb & hsv
## vgg19
默认使用jittor
大约1s能完成
## vgg19_tensorflow
配置虚拟环境：
```bash
conda create -n tf-gpu tensorflow-gpu=2.1.0
conda activate tf-gpu
```
速度：NVIDIA TITAN RTX上，大约1s能完成2~3次迭代。