# resnet18 on imagenet
DATAPATH="data/ImageNet"
python main_imagenet.py \
    --data_path $DATAPATH \
    --arch resnet18 \
    --warkers 8 \
    --n_bits_w 2 \
    --channel_wise \
    --n_bits_a 4 \
    --act_quant \
    --num_samples 128 \
    #--test_before_calibration | tee resnet18_imagenet.log