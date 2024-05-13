# resnet18 on imagenet
DATAPATH="data/ImageNet"
ARCH="resnet18"
n_bits_w=4
batch_size=64
num_samples=1024

for iter in {20000,}; do
    echo "Arch $ARCH"
    echo "W$n_bits_w""A32"
    echo "Iteration $i"
    echo "n_bit_w $n_bits_w"
    echo "batch_size $batch_size"
    python main_imagenet.py \
    --data_path $DATAPATH \
    --arch $ARCH \
    --test_before_calibration \
    --n_bits_w $n_bits_w \
    --channel_wise \
    --iters_w $iter \
    --batch_size $batch_size \
    --num_samples $num_samples\
    --workers 8 \
    | tee $ARCH"_iter"$iter"_tmp.log"
    # --act_quant \
    # --n_bits_a 4 \
    # --test_before_calibration \
done