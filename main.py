import pprint
import torch
import wandb
import os
from args_helper import args
from regularize import trainer as regularize_trainer
from logger import setup_logger
from main_utils import set_seed, get_dataset, get_model, get_optimizer, get_scheduler, update_iter_and_epochs, get_criterion


def main():
    logger = setup_logger(name=args.logger_name, args=args)
    if args.load_pretrained_model:
        logger.info("Resuming Training:")
    else:
        logger.info("Call with args: \n{}".format(pprint.pformat(vars(args))))

    if args.arch == 'shallow_NN':
        project_name = "{}_{}_width_{}".format(args.which_dataset.lower(), args.arch.lower(), args.num_hidden)
    else:
        project_name = "{}_{}".format(args.which_dataset.lower(),args.arch.lower())
    
    if args.label_corruption != 0:
        project_name = project_name + "_label_corruption"
    id = wandb.util.generate_id()
    if args.load_pretrained_model:
        results_dict = torch.load(os.path.join(args.results_dir, "result.pt"))
        id = results_dict['wandb_id']
    wandb.init(project=project_name, entity="jshenouda", name=args.logger_name, config=vars(args), id=id, resume="allow")
    
    if wandb.run.resumed:
        checkpoint = torch.load(wandb.restore(args.checkpoint_path).name)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        set_seed(args.seed, logger)

    if args.cuda:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
        torch.set_num_threads(4)
    logger.info("Using device {}".format(device))
    dataset = get_dataset(args=args, logger=logger)
    update_iter_and_epochs(dataset=dataset, args=args, logger=logger)
    model = get_model(args=args, logger=logger, dataset=dataset).to(device)
    criterion = get_criterion(criterion_type=args.criterion)
    optimizer = get_optimizer(args=args, model=model)
    scheduler = get_scheduler(optimizer=optimizer, logger=logger, args=args)

    if args.regularize:
        _, _ = regularize_trainer(
            dataset=dataset, device=device, model=model,
            args=args, optimizer=optimizer, scheduler=scheduler, criterion=criterion, logger=logger, wandb=wandb)

if __name__ == "__main__":
    main()
