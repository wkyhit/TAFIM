import os
import pprint
import torch
import cv2
from tqdm import tqdm
from configs import data_config
import torchvision.utils as vutils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.dataset import ImagesDataset
from model_archs.simswap.fs_model import fsModel
from utils.image_utils import create_target_img, tensor2im
from configs.transforms_config import adv_img_transform
from configs.paths_config import SIMSWAP_ATTACK_BASE_DIR, ATTACK_BASE_DIR
from configs.common_config import device, dataset_type, val_imgs, resize_size
from model_archs.protection_net.networks import define_G as GenPix2Pix
from configs.attack_config import no_dropout, init_type, init_gain, ngf, net_noise, norm
from configs.transforms_config import img_transform_simswap

if __name__ == '__main__':
    config_parser = ArgumentParser()
    config_parser.add_argument('-p', '--SimSwap_protection_path', default='unet_64/SimSwap_protection_unet_64_10perturb_latest.pth', type=str, help='Path to SimSwap pretrained model')
    config_parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('-l', '--loss_type', default='l2', type=str, help='Loss Type')
    opts = config_parser.parse_args()

    SimSwap_protection_path = opts.SimSwap_protection_path
    loss_type = opts.loss_type
    batch_size = opts.batch_size

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    dataset_args = data_config.DATASETS[dataset_type]
    transforms_dict = dataset_args['transforms']().get_transforms()

    test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'], target_root=dataset_args['test_target_root'], source_transform=transforms_dict['transform_inference'], target_transform=transforms_dict['transform_inference'], num_imgs=val_imgs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    checkpoint = torch.load(os.path.join(SIMSWAP_ATTACK_BASE_DIR, SimSwap_protection_path))
    simswap_net = fsModel(perturb_wt=checkpoint['perturb_wt'], attack_loss_type=loss_type)

    # pSp_net = pSp(perturb_wt=checkpoint['perturb_wt'], attack_loss_type=loss_type)
    print("perturb Weight", checkpoint['perturb_wt'])
    simswap_net.eval().to(device)

    protection_model = GenPix2Pix(6, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    protection_model.load_state_dict(checkpoint['protection_net'])
    protection_model.to(device)
    protection_model.eval()
    global_adv_noise = checkpoint['global_noise'].to(device)

for idx, data in enumerate(tqdm(test_loader, desc='')):  # inner loop within one epoch
        with torch.no_grad():

            simswap_net.real_A1 = data['A'].to(device).clone().detach() #target image
            simswap_net.real_A2 = data['B'].to(device).clone().detach() #source image
            b_size = simswap_net.real_A1.shape[0]
            # Ideal output to produce
            simswap_net.y = create_target_img(batch_size=b_size, size=resize_size, img_transform=img_transform_simswap, color=(255, 0, 0)).to(device)

            # Get the old output
            old_out = simswap_net.swap_face(simswap_net.real_A1, simswap_net.real_A2)

            adv_input = torch.cat((simswap_net.real_A1, global_adv_noise.unsqueeze(0)), 1)
            adv_noise = protection_model(adv_input)
            simswap_net.adv_A1 = torch.clamp(simswap_net.real_A1 + adv_noise, 0, 1)  # For Visualization

            #  Redo Forward pass on adversarial image through the model
            # fake_B为攻击后生成的图片
            simswap_net.fake_B = simswap_net.swap_face(simswap_net.adv_A1, simswap_net.real_A2)

            # calculate loss
            # simswap_net.closure(model_adv_noise=adv_noise, global_adv_noise=global_adv_noise)

            # pSp_net.fake_B = pSp_net(pSp_net.adv_A)
            visuals = simswap_net.get_current_visuals()  # get image results

            img_keys = list(visuals.keys())
            # 分别为：ori_target, adv_target, noise, adv_output, ori_output, ori_source
            images = torch.cat((simswap_net.real_A1, simswap_net.adv_A, simswap_net.adv_A - simswap_net.real_A1, visuals['fake_B'], old_out, simswap_net.real_A2), -1)
            horizontal_grid = tensor2im(vutils.make_grid(images))
            horizontal_grid = cv2.cvtColor(horizontal_grid, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(ATTACK_BASE_DIR, 'visuals', f"{idx}.png"), horizontal_grid)
