import argparse
import os.path
import time
import cv2
import pandas as pd
import torch.cuda
import numpy as np
from models.demo import FPC
import utils
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from lpips.lpips import *

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.25, type=float)
parser.add_argument("--device", default="0")
opt = parser.parse_args()
opt.device = "cuda:0"

def testing(network, save_img):
    datasets = ["Set11_GREY", "Set14_GREY", "Urban100_GREY", "General100_GREY"]
    time_all = []
    total_time = 0.
    lpips_model = LPIPS(net='vgg').to(config.device)
    for idx, item in enumerate(datasets):
        sum_psnr, sum_ssim = 0., 0.
        sum_lpips = 0.
        i = 0
        path = os.path.join('/home/wcr/WXY/dataset/PNG/Grey', item)
        # path = os.path.join('E:/dataset/RGB', item)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        with torch.no_grad():
            for root, dir, files in os.walk(path):
                for file in files:
                    i = i + 1
                    name = file.split('.')[0]
                    Img = cv2.imread(f"{root}/{file}")
                    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                    Img_rec_yuv = Img_yuv.copy()
                    Iorg_y = Img_yuv[:, :, 0]
                    x = Iorg_y / 255.
                    x = torch.from_numpy(np.array(x)).float()

                    h, w = x.size()
                    lack = config.block_size - h % config.block_size if h % config.block_size != 0 else 0
                    padding_h = torch.zeros(lack, w)
                    expand_h = h + lack
                    inputs = torch.cat((x, padding_h), 0)
                    lack = config.block_size - w % config.block_size if w % config.block_size != 0 else 0
                    expand_w = w + lack
                    padding_w = torch.zeros(expand_h, lack)
                    inputs = torch.cat((inputs, padding_w), 1).unsqueeze(0).unsqueeze(0)
                    inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.block_size, dim=3), dim=0)
                    inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.block_size, dim=2), dim=0).to(config.device)
            
                    reconstruction = network(inputs)

                    idx = expand_w // config.block_size
                    reconstruction = torch.cat(torch.split(reconstruction, split_size_or_sections=1 * idx, dim=0), dim=2)
                    reconstruction = torch.cat(torch.split(reconstruction, split_size_or_sections=1, dim=0), dim=3)
                    reconstruction = reconstruction.squeeze()[:h, :w]

                    x_hat = reconstruction.cpu().numpy()

                    psnr = PSNR(x_hat * 255, Iorg_y.astype(np.float64), data_range=255)
                    ssim = SSIM(x_hat * 255, Iorg_y.astype(np.float64), data_range=255)

                    tensor = torch.Tensor(Iorg_y).to(config.device)
                    lpips = lpips_model(reconstruction * 255, tensor)

                    sum_psnr += psnr
                    sum_ssim += ssim
                    sum_lpips += lpips.item()

                    Img_rec_yuv[:,:,0] = x_hat * 255
                    im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                    error = im_rec_rgb - Img
                    error_image_mapped = cv2.applyColorMap(error.astype(np.uint8), cv2.COLORMAP_JET)

                    if save_img:
                        img_path = "./recon_img/Y/{}/{}/".format(item, int(config.ratio * 100))
                        if not os.path.isdir("./recon_img/Y/{}/".format(item)):
                            os.mkdir("./recon_img/Y/{}/".format(item))
                        if not os.path.isdir(img_path):
                            os.mkdir(img_path)
                            print("\rMkdir {}".format(img_path))
                        cv2.imwrite(f"{img_path}/{name}_{round(psnr, 2)}_{round(ssim, 4)}.png", (im_rec_rgb))
                        cv2.imwrite(f"{img_path}/{name}_{round(psnr, 2)}_{round(ssim, 4)}_error1.png",
                                    (error_image_mapped))
                        cv2.imwrite(f"{img_path}/{name}_{round(psnr, 2)}_{round(ssim, 4)}_error2.png", (error))

            print(f"{i} AVG RES: PSNR, {round(sum_psnr / i, 2)}, SSIM, {round(sum_ssim / i, 4)}, LPIPS, {round(sum_lpips / i, 4)}")

        avg = total_time / 100

if __name__=="__main__":
    print("Start evaluate...")
    config = utils.GetConfig(ratio=opt.rate, device=opt.device)
    net = FPC(LayerNo=10, cs_ratio=opt.rate).to(config.device).eval()
    if os.path.exists(config.model):
        if torch.cuda.is_available():
            trained_model = torch.load(config.model, map_location=config.device)
        else:
            trained_model = torch.load(config.model, map_location="cpu")
        net.load_state_dict(trained_model)
        print("Trained model loaded.")
    else:
        raise FileNotFoundError("Missing trained models.")

    testing(net, save_img = config.save)
