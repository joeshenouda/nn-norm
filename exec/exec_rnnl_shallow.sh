# start weight decay vs. path norm regularization
n_gpu=1
config_name="./configs/shallow_rnnl.yml"
lr=0.1
dest_dir="./results/RNNL_shallow"

#-------------------------------------------# w2w1-Prox
thr=0.0003
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
#
thr=0.0002
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
#
thr=0.0001
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
#
thr=0.00005
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &

#------------------------------------# w2v2-prox
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
