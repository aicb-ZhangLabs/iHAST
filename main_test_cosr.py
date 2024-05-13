import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from data.select_dataset import define_Dataset
from models.select_model import define_Model
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score


def main():
    parser = argparse.ArgumentParser(description='Test model with PSNR, SSIM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--json_path', type=str, default='options/test_msrresnet.json', help='Path to option JSON file.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    args = parser.parse_args()

    # Load options
    opt = option.parse(args.json_path, is_train=False)
    opt['dist'] = parser.parse_args().dist
    print(opt['dist'])
    opt['path']['pretrained_netG'] = args.model_path
    opt = option.dict_to_nonedict(opt)

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # Initialize model
    model = define_Model(opt)
    model.init_train()

    # Load pretrained model
    model.load()

    # Create dataset and dataloader
    dataset_opt = opt['datasets']['test']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    # Test
    psnrs, pccs, pcc_cells, mses, ssim = [], [], [], [], []

    for idx, test_data in enumerate(test_loader):
        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])

        # Calculate metrics
        current_psnr = util.calculate_psnr(E_img, H_img, border=opt['scale'])
        current_pcc = util.calculate_pcc(E_img, H_img)
        current_mse = mse(E_img, H_img)
        current_ssim = util.calculate_ssim(E_img, H_img, border=opt['scale'])
        # current_ssim = ssim(E_img, H_img, data_range=1, multichannel=True)

        psnrs.append(current_psnr)
        pccs.append(current_pcc[0])
        pcc_cells.append(current_pcc[1])
        mses.append(current_mse)
        ssim.append(current_ssim)
        # save the E_img with H_img to visualize the results
        # np.save(os.path.join('image/dpsr', f'{idx}_E.npy'), E_img)
        # np.save(os.path.join('image/dpsr', f'{idx}_H.npy'), H_img)

    avg_psnr = np.mean(psnrs)
    avg_mse = np.mean(mses)
    avg_pcc = np.mean(pccs)
    avg_pcc_cell = np.nanmean(pcc_cells)
    avg_ssim = np.nanmean(ssim)

    print(f"Average: PSNR : {avg_psnr:.2f} dB, PCC : {avg_pcc:.4f}, mse : {avg_mse:.4f}," 
          f"PCC_cell : {avg_pcc_cell:.4f}, SSIM : {avg_ssim:.4f}")

if __name__ == '__main__':
    main()
