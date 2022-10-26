CUDA_VISIBLE_DEVICES="-1" python vgg_comp.py \
    --ref /data/chentuo/jittor/data/train_resized/imgs \
    --gen /data/chentuo/jittor/jittor-ThisNameIsGeneratedByJittor-Landscape/SPADE-jittor/results/bs4vae/test_90/images/synthesized_image \
    --vgg_path /data/chentuo/jittor/fast-style-transfer/data/imagenet-vgg-verydeep-19.mat.1 > wRef.out

CUDA_VISIBLE_DEVICES="-1" python vgg_comp.py \
    --ref /data/chentuo/jittor/data/train_resized/imgs \
    --gen /data/chentuo/jittor/jittor-ThisNameIsGeneratedByJittor-Landscape/SPADE-jittor/results/bs4vae/test_90_withoutref/images/synthesized_image \
    --vgg_path /data/chentuo/jittor/fast-style-transfer/data/imagenet-vgg-verydeep-19.mat.1 > woRef.out
