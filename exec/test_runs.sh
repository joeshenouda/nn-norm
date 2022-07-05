n_gpu=0
config_name="./configs/shallow_mnist_subset.yml"
lr=0.1
dest_dir="./results/MNIST_shallow_subset"

##---Iterations---##
total_iter=3
log_freq=1
save_freq=1
##-------Local Test Runs-------#
thr=0.0003
logger_name="wd_{$thr}_{$lr}"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --total-iter $total_iter --iter-period $total_iter --debias-iter $total_iter --save-freq $save_freq --log-freq $log_freq --algo v0 --threshold $thr --lr $lr --regularize --with-loss-term --loss-term wd --w-norm-degree 2 --v-norm-degree 1
