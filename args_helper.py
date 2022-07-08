import argparse
import yaml
import sys
import traceback

from configs import parser as _parser

global args
#print(sys.modules)
#traceback.print_stack()
class ArgsHelper:
    def parse_arguments(self, jupyter_mode=False):
        parser = argparse.ArgumentParser(description="Neural Network Norms")

        # ============================================================================================================ #
        # system related
        parser.add_argument(
            "--cuda",
            action="store_true",
            required=False,
            default=False,
            help="[system] bool value to use gpu or not"
        )
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            required=False,
            default=False,
            help="[system] bool value to use no cuda or not"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            metavar="G",
            help="[system] Override the default choice for a CUDA-enabled GPU by specifying the GPU\"s integer index "
                 "(i.e. \"0\" for \"cuda:0\") "
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            metavar="W",
            help="[system] Number of workers"
        )
        parser.add_argument(
            "--logger-name",
            type=str,
            required=True,
            help="[system] logger name needs to be specified"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="[system] Set seed for the program"
        )
        parser.add_argument(
            "--dest-dir",
            type=str,
            default=None,
            help="[system] result destination directory, will be overwritten later"
        )
        parser.add_argument(
            "--results-dir",
            type=str,
            default=None,
            help="[system] the directory for results, used for loading pretrained model"
        )
        parser.add_argument(
            "--config",
            default=None,
            help="Config file to use"
        )
        # ============================================================================================================ #
        # data related
        parser.add_argument(
            "--data-path",
            default="data/",
            help="[dataset] path to dataset base directory"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help="[dataset] Input batch size for training (default: 64)"
        )
        parser.add_argument(
            "--samps-per-class",
            type=int,
            default=100,
            help="[dataset] number of samples per class, used for subsampling MNIST"
        )
        parser.add_argument(
            "--which-dataset",
            type=str,
            default="mnist",
            help="[dataset] Dataset to train the model with. Can be mnist or random (default: mnist)"
        )
        parser.add_argument(
            "--random-dim",
            type=int,
            default=700,
            help="[dataset] dimensions for random dataset"
        )
        parser.add_argument(
            "--rnnl-samples",
            type=int,
            default=100,
            help="[data] Num samples for RNNL dataset"
        )
        parser.add_argument(
            "--rnnl-dim",
            type=int,
            default=10,
            help="[data] Input dimension for RNNL dataset"
        )
        parser.add_argument(
            "--rnnl-neurons",
            type=int,
            default=100,
            help="[data] Number of neurons for RNNL dataset"
        )
        parser.add_argument(
            "--rnnl-out-dim",
            type=int,
            default=10,
            help="[data] Dimension for labels of RNNL dataset"
        )
        parser.add_argument(
            "--flip-one-hot",
            action="store_true",
            required=False,
            default=False,
            help="[data] For binary experiments this will flip the one-hot encoding of labels"
        )
        parser.add_argument(
            "--synth-labels",
            action="store_true",
            required=False,
            default=False,
            help="[dataset] use synthetic labels for binary MNIST"
        )
        # augmentation
        parser.add_argument(
            '--smoothing',
            type=float,
            default=0.1,
            help='[dataset] Label smoothing (default: 0.1)'
        )
        parser.add_argument(
            '--transfer-learning',
            default=False,
            action="store_true",
            help="[dataset] if set to true, then it is transfer learning"
        )
        # ============================================================================================================ #
        # optimizer related
        parser.add_argument(
            "--optimizer",
            type=str,
            default="SGD",
            help="[optimizer] optimizer, choice  SGD | Adam | AdamW"
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="[optimizer] learning rate"
        )
        parser.add_argument(
            "--wd",
            type=float,
            default=0.0001,
            help="[optimizer] weight decay"
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="[optimizer] momentum"
        )
        # ============================================================================================================ #
        # criterion related
        parser.add_argument(
            "--criterion",
            type=str,
            default="CE",
            help="[optimizer] optimizer, choice  CE | MSE | BCE"
        )
        # ============================================================================================================ #
        # scheduler related
        parser.add_argument(
            "--lr-scheduler",
            type=str,
            default=None,
            help="[scheduler] learning rate scheduler"
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=None,
            help="[scheduler] learning rate decay"
        )
        parser.add_argument(
            "--milestone",
            type=int,
            nargs="+",
            default=[],
            help="[scheduler] milestone for multi-step learning rate scheduler"
        )
        parser.add_argument(
            '--warmup-lr',
            type=float,
            default=1e-6,
            metavar='LR',
            help='[scheduler] warmup learning rate (default: 1e-6)'
        )
        parser.add_argument(
            '--min-lr',
            type=float,
            default=1e-5,
            metavar='LR',
            help='[scheduler] lower lr bound for cyclic schedulers that hit 0 (1e-5)'
        )
        parser.add_argument(
            '--warmup-epochs',
            type=int,
            default=5,
            metavar='N',
            help='[scheduler] epochs to warmup LR, if scheduler supports'
        )
        # ============================================================================================================ #
        # model related
        parser.add_argument(
            "--arch",
            type=str,
            default="[model] architecture name, choice: Lenet",
        )
        parser.add_argument(
            "--drop-path",
            type=float,
            default=0.1,
            help="[model] Drop path rate (default: 0.1)",
        )
        parser.add_argument(
            "--act-fn",
            type=str,
            default="relu",
            help="[model] the activation function, choice: relu | gelu"
        )
        parser.add_argument(
            "--load-pretrained-model",
            action="store_true",
            default=False,
            help="[model] load the pretrained model"
        )
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help="[model] checkpoint path for model"
        )
        parser.add_argument(
            "--two-layer-classifier",
            action="store_true",
            default=False,
            help="[model] use two linear layer with relu in between as the final classifier"
        )
        parser.add_argument(
            "--num-hidden",
            type=int,
            default=None,
            help="[model] indicate number of hidden neurons "
        )
        # ============================================================================================================ #
        # Mixup params
        parser.add_argument('--mixup-flag', default=False, action="store_true",
                            help="turn on or off the mixup")
        parser.add_argument('--mixup', type=float, default=0.8,
                            help='[mixup] mixup alpha, mixup enabled if > 0. (default: 0.8)')
        parser.add_argument('--cutmix', type=float, default=1.0,
                            help='[mixup] cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
        parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                            help='[mixup] cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
        parser.add_argument('--mixup-prob', type=float, default=1.0,
                            help='[mixup] Probability of performing mixup or cutmix when either/both is enabled')
        parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                            help='[mixup] Probability of switching to cutmix when both mixup and cutmix enabled')
        parser.add_argument('--mixup-mode', type=str, default='batch',
                            help='[mixup] How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
        # ============================================================================================================ #
        # training related
        parser.add_argument(
            "--total-iter",
            type=int,
            default=0,
            help="[train] total number of iterations to run"
        )
        parser.add_argument(
            "--total-epoch",
            type=int,
            default=0,
            help="[train] total number of iterations to run"
        )
        parser.add_argument(
            "--iter-period",
            type=int,
            default=None,
            help="[train] the period consist of training | prune | debias and train"
        )
        parser.add_argument(
            "--prune-iter",
            type=int,
            default=None,
            help="[train] in the iter_period, begin which iteration, starts to prune"
        )
        parser.add_argument(
            "--debias-iter",
            type=int,
            default=None,
            help="[train] in the iter_period, begin which iteration, stop to prune"
        )
        parser.add_argument(
            "--log-freq",
            type=int,
            help="[train] frequency to print out the test and val information"
        )
        parser.add_argument(
            "--save-freq",
            type=int,
            help="[train] frequency to save the model and loss information"
        )
        # ============================================================================================================ #
        # hyperparameter related
        parser.add_argument(
            "--threshold",
            type=float,
            help="[hyperparameter] threshold in the soft-threshold pruning"
        )
        # ============================================================================================================ #
        # algorithm related
        parser.add_argument(
            "--weight-training",
            action="store_true",
            default=False,
            help="[algorithm] if set true, only train, without pruning"
        )

        parser.add_argument(
            "--only-balance",
            action="store_true",
            default=False,
            help="[algorithm] if set true, only train and do weight balancing"
        )
        parser.add_argument(
            "--pretrained",
            type=str,
            default=None,
            help="[algorithm] if set true, load the pretrained model and start finetuning"
        )
        parser.add_argument(
            "--algo",
            type=str,
            default='v0',
            help="[algorithm] which pruning algorithm to use, choice are v0 | v1"
        )
        parser.add_argument(
            "--pf-lr-factor",
            type=int,
            help="[algorithm] the low rank factor for Pufferfish"
        )
        parser.add_argument(
            "--small-network-nact1",
            type=int,
            default=None,
            help="[algorithm] the number of neurons in small network layer 1"
        )
        parser.add_argument(
            "--small-network-nact2",
            type=int,
            default=None,
            help="[algorithm] the number of neurons in small network layer 2"
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=None,
            help="[algorithm] the number of samples, now only used in synthetic data"
        )
        # ============================================================================================================ #
        # regularize related
        parser.add_argument(
            "--regularize",
            action="store_true",
            default=False,
            help="[regularize] if set true, simply regularize the path norm, but not prune the path norm"
        )
        parser.add_argument(
            "--with-loss-term",
            action="store_true",
            default=False,
            help="[regularize] if set true, regularize with some term add to loss"
        )
        parser.add_argument(
            "--loss-term",
            default=None,
            type=str,
            help="[regularize] choice: wd | pn"
        )
        parser.add_argument(
            "--with-prox-upd",
            action="store_true",
            default=False,
            help="[regularize] if set true, regularize the coupling layers with path norm, the other part with some term add to loss"
        )
        parser.add_argument(
            "--w-norm-degree",
            type=int,
            default=2,
            help="[regularize] path norm degree for w norm"
        )
        parser.add_argument(
            "--v-norm-degree",
            type=int,
            default=1,
            help="[regularize] path norm degree for v norm"
        )
        parser.add_argument(
            "--plus",
            action="store_true",
            default=False,
            help="[regularize] if set to true, will plus the path norm term instead of multiply"
        )
        parser.add_argument(
            "--layerwise-balance",
            action="store_true",
            default=False,
            help="[regularize] if set true, even doing sgd update, still balance the v layerwise"
        )
        # ============================================================================================================ #
        # corruption on labels
        parser.add_argument(
            "--label-corruption",
            type=float,
            default=0,
            help="[corruption] range in [0, 1]. If 0, no corruption on label. If 1, all labels are corrupted."
        )

        if jupyter_mode:
            args = parser.parse_args("")
        else:
            args = parser.parse_args()
        self.get_config(args, jupyter_mode)

        return args

    def get_config(self, parser_args, jupyter_mode=False):
        # get commands from command line
        override_args = _parser.argv_to_vars(sys.argv)

        # load yaml file
        if parser_args.config is not None:
            yaml_txt = open(parser_args.config).read()

            # override args
            loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
            if not jupyter_mode:
                for v in override_args:
                    loaded_yaml[v] = getattr(parser_args, v)
            #import ipdb; ipdb.set_trace()

            print(f"=> Reading YAML config from {parser_args.config}")
            parser_args.__dict__.update(loaded_yaml)

    def isNotebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    def get_args(self):
        global args
        jupyter_mode = self.isNotebook()
        args = self.parse_arguments(jupyter_mode)
        from main_utils import set_dest_dir
        if args.dest_dir is None:
            set_dest_dir(args)


argshelper = ArgsHelper()
argshelper.get_args()
