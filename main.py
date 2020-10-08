import os
import sys
import logging

import torch
import numpy as np

from models import c11
from src.attacks import pgd_rand
from src.train import train_standard, train_adv
from src.evaluation import test_clean, test_adv
from src.args import get_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint
from src.utils_general import seed_everything, get_model, get_optim

def train(args, epoch, logger, loader, model, opt, device):
    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(logger, epoch, loader, model, opt, device)

    elif args.method == "adv":
        train_log = train_adv(logger, epoch, loader, args.pgd_steps, model, opt, device)

    else:
        raise  NotImplementedError("Training method not implemented!")

    logger.add_scalar("train/acc_ep", train_log[0], epoch+1)
    logger.add_scalar("train/loss_ep", train_log[1], epoch+1)
    logging.info(
        "Epoch: [{0}]\t"
        "Loss: {loss:.6f}\t"
        "Accuracy: {acc:.2f}".format(
            epoch,
            loss=train_log[1],
            acc=train_log[0]))

    return train_log

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    attack_param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 20, "restart": 1}

    args = get_args()
    logger = metaLogger(args)
    logging.basicConfig(
        filename=args.j_dir+ "/log/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader = load_dataset(args.dataset, args.batch_size)

    model = get_model(args, device)
    opt, lr_scheduler = get_optim(model, args)
    ckpt_epoch = 0
    
    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    ckpt_location = os.path.join(ckpt_dir, "custome_ckpt_"+logger.ckpt_status+".pth")
    if os.path.exists(ckpt_location):
        ckpt = torch.load(ckpt_location)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        ckpt_epoch = ckpt["epoch"]
        if lr_scheduler:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        print("LOADED CHECKPOINT")

    for _epoch in range(ckpt_epoch, args.epoch):
        train_log = train(args, _epoch, logger, train_loader, model, opt, device)

        test_log = test_clean(test_loader, model, device)
        adv_log = test_adv(test_loader, model, pgd_rand, attack_param, device)

        logger.add_scalar("pgd20/acc", adv_log[0], _epoch+1)
        logger.add_scalar("pgd20/loss", adv_log[1], _epoch+1)
        logger.add_scalar("test/acc", test_log[0], _epoch+1)
        logger.add_scalar("test/loss", test_log[1], _epoch+1)
        logging.info(
            "Test set: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=test_log[1],
                acc=test_log[0]))
        logging.info(
            "PGD20: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=adv_log[1],
                acc=adv_log[0]))

        if lr_scheduler:
            lr_scheduler.step()

        if (_epoch+1) % args.ckpt_freq == 0:
            rotateCheckpoint(ckpt_dir, "custome_ckpt", model, opt, _epoch, lr_scheduler)

        logger.save_log()
    logger.close()
    torch.save(model.state_dict(), args.j_dir+"/model/model.pt")

if __name__ == "__main__":
    main()
