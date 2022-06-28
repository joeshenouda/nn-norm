# -------------------------------------------------------------------------------------------------------------------- #
# start weight decay vs. path norm regularization
n_gpu=0
config_name="./configs/shallow_mnist_subset.yml"
lr=0.1
dest_dir="./results/MNIST_shallow_subset"
##--------Save model test------#
thr=0.001
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 --load-pretrained-model --checkpoint-path "results/MNIST_subset_shallow_NN/0626113416_pn_w2v1_loss_{0.001}_{0.1}_balance/model_idx_6800_acc_74_14_sp_1445.pt" --results-dir "./results/MNIST_subset_shallow_NN/0626113416_pn_w2v1_loss_{0.001}_{0.1}_balance/" &

# --------------------------------------------# wd experiments
thr=0.0003
logger_name="wd_{$thr}_{$lr}"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-loss-term --loss-term wd --w-norm-degree 2 --v-norm-degree 1 &

#-------------------------------------------# w2w1-Prox
thr=0.0003
logger_name="pn_w2v1_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 1 &
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
#------------------------------------# w2v2-prox
thr=0.0003
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --threshold $thr --lr $lr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 --load-pretrained-model --checkpoint-path "results/MNIST_subset_shallow_NN/0620212437_pn_w2v2_loss_{0.0003}_{0.1}_balance/model_idx_10000_acc_86_48_sp_205.pt" --results-dir "./results/MNIST_subset_shallow_NN/0620212437_pn_w2v2_loss_{0.0003}_{0.1}_balance" &

thr=0.0003
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &

thr=0.0004
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &

thr=0.0005
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &

thr=0.001
logger_name="pn_w2v2_loss_{$thr}_{$lr}_balance"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-prox-upd --w-norm-degree 2 --v-norm-degree 2 &
#------------------------------------# w2v2-SGD
thr=0.00005
logger_name="pn_w2v2_loss_{$thr}_{$lr}_SGD"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-loss-term --loss-term pn --w-norm-degree 2 --v-norm-degree 2 &

thr=0.00003
logger_name="pn_w2v2_loss_{$thr}_{$lr}_SGD"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-loss-term --loss-term pn --w-norm-degree 2 --v-norm-degree 2 &

thr=0.00001
logger_name="pn_w2v2_loss_{$thr}_{$lr}_SGD"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold $thr --regularize --with-loss-term --loss-term pn --w-norm-degree 2 --v-norm-degree 2 &

#------------------------------------# vanilla no reg
logger_name="unreg_{$lr}"
#python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --algo v0 --lr $lr --threshold 0 --regularize --w-norm-degree 2 --v-norm-degree 1 &

