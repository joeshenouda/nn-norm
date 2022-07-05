# -------------------------------------------------------------------------------------------------------------------- #
# start weight decay vs. path norm regularization
n_gpu=1
config_name="./configs/shallow_binary.yml"
lr=0.1
dest_dir="./results/MNIST_shallow_binary"
##--------Save model test------#
thr=0.001   
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --load-pretrained-model --checkpoint-path "results/MNIST_binary_shallow_NN/0616140133_pn_w2v1_loss_{0.001}_{0.1}_balance/model_idx_4500_acc_99_72_sp_82.pt" --results-dir "./results/MNIST_binary_shallow_NN/0616140133_pn_w2v1_loss_{0.001}_{0.1}_balance" &

thr=0.001   
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --load-pretrained-model --checkpoint-path "results/MNIST_binary_shallow_NN/0616140133_pn_w2v2_loss_{0.001}_{0.1}_balance/model_idx_4500_acc_99_53_sp_65.pt" --results-dir "./results/MNIST_binary_shallow_NN/0616140133_pn_w2v2_loss_{0.001}_{0.1}_balance" &

##------Exact copy of flipped one-hot encoding but without the flip------#
thr=0.0003
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --total-iter 20000 --iter-period 20000 --debias-iter 20000 --log-freq 20 --save-freq 2000 &

# --------------------------------------------# wd experiments
thr=0.001
logger_name="wd_{$thr}_{$lr}"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-loss-term --loss-term wd --w-norm-degree 2 --v-norm-degree 1 &

#-------------------------------------------# w2w1-Prox
thr=0.0003
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1
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

#---------------------------------------# w2v1-SGD
thr=0.02
logger_name="pn_w2v1_loss_{$thr}_SGD"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --regularize --with-loss-term --loss-term pn --w-norm-degree 2 --v-norm-degree 1
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

#------------------------------------# w2v2-SGD
thr=0.00005
logger_name="pn_w2v2_loss_{$thr}_{$lr}_SGD"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-loss-term --loss-term pn --w-norm-degree 2 --v-norm-degree 2 &

#------------------------------------# vanilla no reg
logger_name="unreg_{$lr}"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold 0 --regularize --w-norm-degree 2 --v-norm-degree 1 &

