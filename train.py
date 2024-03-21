import os
import sys
import time
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import scipy.io as scio
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from models.demo import FPC
import models
import utils
from skimage.metrics import peak_signal_noise_ratio as PSNR
from utils.scheduler import *
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.10, type=float)
parser.add_argument('--start_epoch', default=0, type=int, help='epoch number of start training')
parser.add_argument('--warm_epochs', default=3, type=int, help='number of epochs to warm up')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning_rate', default=2e-4, type=float, help='initial learning rate')
parser.add_argument('--last_lr', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of the Net')
parser.add_argument("--bs", default=64, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
opt = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val_p(config, net):
    net = net.eval()
    file_no = [11]
    folder_name = ["SET11"]

    for idx, item in enumerate(folder_name):
        p_total = 0
        path = "{}/".format(config.val_path) + item
        with torch.no_grad():
            for i in range(file_no[idx]):
                name = "{}/({}).mat".format(path, i + 1)
                x = scio.loadmat(name)['temp3']
                x = torch.from_numpy(np.array(x)).to(config.device)
                x = x.float()
                ori_x = x

                h, w = x.size()
                lack = config.block_size - h % config.block_size if h % config.block_size != 0 else 0
                padding_h = torch.zeros(lack, w).to(config.device)
                expand_h = h + lack
                inputs = torch.cat((x, padding_h), 0)

                lack = config.block_size - w % config.block_size if w % config.block_size != 0 else 0
                expand_w = w + lack
                padding_w = torch.zeros(expand_h, lack).to(config.device)
                inputs = torch.cat((inputs, padding_w), 1).unsqueeze(0).unsqueeze(0)

                inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.block_size, dim=3), dim=0)
                inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.block_size, dim=2), dim=0)

                reconstruction = net(inputs)

                blocks = expand_w // config.block_size
                reconstruction = torch.cat(
                    torch.split(reconstruction, split_size_or_sections=1 * blocks, dim=0), dim=2)
                reconstruction = torch.cat(torch.split(reconstruction, split_size_or_sections=1, dim=0), dim=3)
                reconstruction = reconstruction.squeeze()[:h, :w]

                ori_x = ori_x.cpu().numpy() * 255.
                x_hat = reconstruction.cpu().numpy() * 255.
                x_hat = np.rint(np.clip(x_hat, 0, 255))
                p = PSNR(ori_x, x_hat, data_range=255)

                p_total = p_total + p

            return p_total / file_no[idx]


def main():
    device = "cuda:" + opt.device
    config = utils.GetConfig(ratio=opt.rate, device=device)
    config.check()
    set_seed(22)
    print("Data loading...")
    torch.cuda.empty_cache()
    dataset_train = utils.train_loader(batch_size=opt.bs)

    net = FPC(LayerNo=10, cs_ratio=opt.rate).to(config.device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs - opt.warm_epochs,
                                                            eta_min=opt.last_lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warm_epochs,
                                       after_scheduler=scheduler_cosine)

    if os.path.exists(config.model):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.model, map_location=config.device))
            info = torch.load(config.info, map_location=config.device)
        else:
            net.load_state_dict(torch.load(config.model, map_location="cpu"))
            info = torch.load(config.info, map_location="cpu")

        start_epoch = info["epoch"]
        best = info["res"]
        print("Loaded trained model of epoch {:2}, res: {:8.4f}.".format(start_epoch, best))
    else:
        start_epoch = 1
        scheduler.step()
        best = 0
        print("No saved model, start epoch = 1.")

    print("Sensing Rate: %.2f , Epoch: %d , Initial LR: %f\n" % (opt.rate, opt.epochs, opt.lr))
    loss_record = []
    over_all_time = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        print('current lr {:.5e}'.format(scheduler.get_lr()[0]))
        epoch_loss = 0
        dic = {"rate": config.ratio, "epoch": epoch, "device": config.device}
        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            xi = xi.to(config.device)
            optimizer.zero_grad()
            xo = net(xi)
            batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.device)
            epoch_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}]"
                           .format(config.batch_size * (idx + 1), dataset_train.__len__() * config.batch_size,
                                   batch_loss.item()))

        avg_loss = epoch_loss / dataset_train.__len__()
        loss_record.append(avg_loss)
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]".format(epoch, avg_loss))
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, "%s/net_params_%d.pth" % (config.folder, epoch))

        if epoch == 1:
            if not os.path.isfile(config.log):
                output_file = open(config.log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n".format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

            if not os.path.isfile(config.record):
                record_file = open(config.record, 'w')
                record_file.write("=" * 120 + "\n")
                record_file.close()
            record_file = open(config.log, 'r+')
            oldR = record_file.read()
            record_file.seek(0)
            record_file.write("\nAbove is {} test. Note：{}.\n".format("???", None) + "=" * 120 + "\n")
            record_file.write(oldR)
            record_file.close()

        # print("\rNow val..")
        p = val_p(config, net)
        print_data = "[%02d/%02d]Total Loss: %f, learning_rate: %.5f, Res: %.3f\n" % (
        epoch, opt.epochs, avg_loss, scheduler.get_lr()[0], p)
        print(print_data)
        record_file = open(config.record, 'r+')
        oldr = record_file.read()
        record_file.seek(0)
        record_file.write(print_data)
        record_file.write(oldr)
        record_file.close()
        # print("{:5.3f}".format(p))
        if p > best:
            info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.model)
            torch.save(info, config.info)
            print("*", "  Check point of epoch {:2} saved  ".format(epoch).center(120, "="), "*")
            best = p
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write("Epoch {:2.0f}, Loss of train {:8.10f}, Res {:2.3f}\n".format(epoch, avg_loss, best))
            output_file.write(old)
            output_file.close()

        scheduler.step()
        print("Over all time: {:.3f}s".format(time.time() - over_all_time))
    print("Train end.")


def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(opt.device) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory


if __name__ == "__main__":
    main()
