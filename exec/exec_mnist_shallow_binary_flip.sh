# -------------------------------------------------------------------------------------------------------------------- #
# start weight decay vs. path norm regularization
n_gpu=1
config_name="./configs/shallow_binary.yml"
lr=0.1
dest_dir="./results/MNIST_shallow_binary"

#---------Flip one-hot experiments------------#
thr=0.0004
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --flip-one-hot --total-iter 20000 --iter-period 20000 --debias-iter 20000 --log-freq 20 --save-freq 2000 &

thr=0.0003
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --flip-one-hot --total-iter 20000 --iter-period 20000 --debias-iter 20000 --log-freq 20 --save-freq 2000 &

thr=0.0005
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --flip-one-hot --total-iter 20000 --iter-period 20000 --debias-iter 20000 --log-freq 20 --save-freq 2000 &

thr=0.00005
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --flip-one-hot --total-iter 20000 --iter-period 20000 --debias-iter 20000 --log-freq 20 --save-freq 2000 &
