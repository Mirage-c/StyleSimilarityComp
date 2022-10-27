CUDA_VISIBLE_DEVICES="1" python vgg_comp.py \
    --ref /mnt/disk1/ct/jittor/data/train_resized/imgs \
    --gen /mnt/disk1/ct/jittor/data/wRef/images/synthesized_image \
    --vgg_path /mnt/disk1/ct/jittor/data/imagenet-vgg-verydeep-19.mat.1 > wRef_all.out

CUDA_VISIBLE_DEVICES="1" python vgg_comp.py \
    --ref /mnt/disk1/ct/jittor/data/train_resized/imgs \
    --gen /mnt/disk1/ct/jittor/data/woRef/images/synthesized_image\
    --vgg_path /mnt/disk1/ct/jittor/data/imagenet-vgg-verydeep-19.mat.1 > woRef_all.out
