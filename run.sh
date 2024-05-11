# resnet18 on imagenet
DATAPATH="data/ImageNet"
ARCH="resnet18"
python main_imagenet.py \
    --data_path $DATAPATH \
    --arch $ARCH \
    --n_bits_w 4 \
    --channel_wise \
    --iters_w 20000 \
    --batch_size 64 \
    --num_samples 1024\
    --workers 8 \
    | tee $ARCH"_imagenet_check_code.txt"
    # --act_quant \
    # --n_bits_a 4 \
    # --test_before_calibration \