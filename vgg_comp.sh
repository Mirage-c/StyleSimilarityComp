CUDA_VISIBLE_DEVICES="0" python3.8 vgg_comp.py \
    --ref /mnt/disk1/ct/jittor/data/train_resized/imgs \
    --gen /mnt/disk1/ct/jittor/JGAN/models/gaugan/results/bs4vae/test_90/images/synthesized_image \
    --vgg_path /mnt/disk1/ct/jittor/data/imagenet-vgg-verydeep-19.mat.1

CUDA_VISIBLE_DEVICES="1" python vgg_comp_tf.py \
    --ref /mnt/disk1/ct/jittor/data/train_resized/imgs \
    --gen /mnt/disk1/ct/jittor/JGAN/models/gaugan/results/bs4vae/test_90/images/synthesized_image \
    --vgg_path /mnt/disk1/ct/jittor/data/imagenet-vgg-verydeep-19.mat.1
