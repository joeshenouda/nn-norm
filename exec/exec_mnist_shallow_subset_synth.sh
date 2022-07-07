# -------------------------------------------------------------------------------------------------------------------- #
# start weight decay vs. path norm regularization
n_gpu=1
config_name="./configs/shallow_mnist_subset_synth.yml"
lr=0.01
dest_dir="./results/MNIST_shallow_subset_synth"

thr=0.0003
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
#
thr=0.0004
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
#
thr=0.0005
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
#
thr=0.001
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1
#-------------------------------------------------------------#
thr=0.0003
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &

thr=0.0004
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &

thr=0.0005
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &

thr=0.001
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &
