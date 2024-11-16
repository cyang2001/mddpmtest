import random
import pywt
from regex import F
from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
import matplotlib.pyplot as plt
import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any, List
import torchio as tio
from src.utils.patch_sampling import BoxSampler
from src.utils.generate_noise import gen_noise
from scipy.ndimage import gaussian_filter, uniform_filter
def get_wavelet_components(slice, wavelet='bior1.3', level=1):
    """
    进行小波变换，获取小波系数
    """
    coeffs = pywt.wavedec2(slice, wavelet, level=level)
    return coeffs

def apply_gaussian_filter(component, sigma=5):
    """
    对小波分量应用高斯滤波器
    """
    return gaussian_filter(component, sigma=sigma)

def apply_uniform_filter(component, size=3):
    """
    对小波分量应用均值滤波器
    """
    return uniform_filter(component, size=size)

def apply_filter_to_wavelet(coeffs, low_pass_filter, high_pass_filter, low_pass_args, high_pass_args):
    """
    直接在小波域应用不同的滤波操作
    """
    LL, (LH, HL, HH) = coeffs
    
    # 应用低通滤波器（高斯滤波器）到低频分量
    LL_filtered = low_pass_filter(LL, **low_pass_args)
    
    # 应用高通滤波器（均值滤波器）到高频分量
    LH_filtered = high_pass_filter(LH, **high_pass_args)
    HL_filtered = high_pass_filter(HL, **high_pass_args)
    HH_filtered = high_pass_filter(HH, **high_pass_args)

    return LL_filtered, LH_filtered, HL_filtered, HH_filtered

def plot_wavelet_components(LL, LH, HL, HH):
    """
    显示小波分量
    """
    titles = ['Approximation (Low-Low)', 'Horizontal detail (Low-High)',
              'Vertical detail (High-Low)', 'Diagonal detail (High-High)']
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    components = [LL, LH, HL, HH]
    
    for i, (component, title) in enumerate(zip(components, titles)):
        ax = axes[i]
        ax.imshow(component, cmap='gray', interpolation='nearest')
        ax.set_title(f'{title} Component')
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.tight_layout()
    # plt.show()
def plot_reconstructed_image(original, reconstructed):
    """
    显示原始图像和重建图像
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def pad_image(image, pad_shape):
    """
    填充图像
    """
    pad_height = pad_shape[0] - image.shape[0]
    pad_width = pad_shape[1] - image.shape[1]
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    return padded_image

def crop_image_to_original(padded_image, original_shape):
    """
    裁剪图像到原始形状
    """
    cropped_image = padded_image[:original_shape[0], :original_shape[1]]
    return cropped_image

def get_freq_image(slice):
    coeffs = get_wavelet_components(slice, wavelet='haar', level=1)
    low_pass_filter = apply_gaussian_filter
    high_pass_filter = apply_uniform_filter
    low_pass_args = {'sigma': 1}
    high_pass_args = {'size': 2}
    LL_filtered, LH_filtered, HL_filtered, HH_filtered = apply_filter_to_wavelet(
    coeffs, low_pass_filter, high_pass_filter, low_pass_args, high_pass_args
    )
    coeffs_filtered = (LL_filtered, (LH_filtered, HL_filtered, HH_filtered))
    reconstructed_image = pywt.waverec2(coeffs_filtered, 'haar', mode='symmetric')
    cropped_reconstructed_image = crop_image_to_original(reconstructed_image, slice.shape)
    return torch.from_numpy(cropped_reconstructed_image).cuda()

def get_freq_image_defaut(slice):
    choice = random.randint(0, 1)

    if (choice == 0):
        val = random.randint(2, 10)

        fft_image = np.fft.fftshift(np.fft.fft2(slice))

        # Create a mask to extract a specific region (e.g., a square in the center)
        mask = np.zeros_like(fft_image)
        rows, cols = fft_image.shape
        center_row, center_col = rows // 2, cols // 2
        mask[center_row - val: center_row + val, center_col - val: center_col + val] = 1

        # Apply the mask to the Fourier transform
        masked_fft = np.multiply(fft_image, mask)

        # Compute the inverse Fourier transform
        reconstructed_image = np.real(np.fft.ifft2(np.fft.ifftshift(masked_fft)))

        # Extract the masked region from the original image
        masked_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image * (1 - mask))))

        # Display the extracted masked region
        # show_two_images_side_by_side(slice2, input[i, 0, :, :])

    else:
        val = random.randint(2, 10)

        fft_image = np.fft.fftshift(np.fft.fft2(slice))

        # Create an inverse mask
        mask = np.zeros_like(fft_image)
        rows, cols = fft_image.shape
        center_row, center_col = rows // 2, cols // 2
        mask[center_row - val: center_row + val, center_col - val: center_col + val] = 1
        inverse_mask = 1 - mask

        # Apply the inverse mask to the Fourier transform
        masked_fft = fft_image * inverse_mask

        # Compute the inverse Fourier transform
        reconstructed_image = np.real(np.fft.ifft2(np.fft.ifftshift(masked_fft)))

        # Extract the masked region from the original image
        masked_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image * mask)))
        # masked_image = masked_image * true_mask_slice_array

    return torch.from_numpy(masked_image).cuda()

def multi_scale_wavelet_transform(image, wavelet='bior1.3', levels=3):
    coeffs = pywt.wavedec2(image, wavelet, level=levels)
    return coeffs

def apply_filters_to_multi_scale_coeffs(coeffs, low_pass_filter, high_pass_filter, low_pass_args, high_pass_args):
    filtered_coeffs = []
    for i in range(len(coeffs)):
        if i == 0:
            filtered_coeffs.append(low_pass_filter(coeffs[i], **low_pass_args))
        else:
            subbands = coeffs[i]
            filtered_subbands = tuple(high_pass_filter(subband, **high_pass_args) for subband in subbands)
            filtered_coeffs.append(filtered_subbands)
    return filtered_coeffs

def get_freq_multi_scale(self, image, wavelet='coif5', levels=2):
    coeffs = multi_scale_wavelet_transform(image, wavelet, levels)
    low_pass_filter = apply_gaussian_filter
    high_pass_filter = apply_uniform_filter
    low_pass_args = {'sigma': 0.5}
    high_pass_args = {'size': 3}
    filtered_coeffs = apply_filters_to_multi_scale_coeffs(coeffs, low_pass_filter, high_pass_filter, low_pass_args, high_pass_args)
    reconstructed_image = pywt.waverec2(filtered_coeffs, wavelet, mode='symmetric')
    cropped_reconstructed_image = crop_image_to_original(reconstructed_image, image.shape)
    return torch.from_numpy(cropped_reconstructed_image).cuda()


def show_two_images_side_by_side(image1, image2):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, current subplot 1
    plt.imshow(image1.cpu().numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, current subplot 2
    plt.imshow(image2.cpu().numpy(), cmap="gray")
    plt.axis("off")

    plt.show()


def show_tensor_image(img):
    # Display the image slice using matplotlib
    plt.imshow(img.cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


class DDPM_2D(LightningModule):
    def __init__(self, cfg, prefix=None):
        super().__init__()

        self.cfg = cfg

        # Model

        model = OpenAI_UNet(
            image_size=(int(cfg.imageDim[0] / cfg.rescaleFactor), int(cfg.imageDim[1] / cfg.rescaleFactor)),
            in_channels=1,
            model_channels=cfg.get('unet_dim', 64),
            out_channels=1,
            num_res_blocks=cfg.get('num_res_blocks', 3),
            attention_resolutions=(
            int(cfg.imageDim[0]) / int(32), int(cfg.imageDim[0]) / int(16), int(cfg.imageDim[0]) / int(8)),
            dropout=cfg.get('dropout_unet', 0),  # default is 0.1
            channel_mult=cfg.get('dim_mults', [1, 2, 4, 8]),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=True,
            use_fp16=True,
            num_heads=cfg.get('num_heads', 4),
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
            use_spatial_transformer=False,
            transformer_depth=1,
        )
        model.convert_to_fp16()

        timesteps = cfg.get('timesteps', 1000)
        self.test_timesteps = cfg.get('test_timesteps', 150)
        sampling_timesteps = cfg.get('sampling_timesteps', self.test_timesteps)

        self.diffusion = GaussianDiffusion(
            model,
            image_size=(int(cfg.imageDim[0] / cfg.rescaleFactor), int(cfg.imageDim[1] / cfg.rescaleFactor)),
            # only important when sampling
            timesteps=timesteps,  # number of steps
            sampling_timesteps=sampling_timesteps,
            objective=cfg.get('objective', 'pred_x0'),  # pred_noise or pred_x0
            channels=1,
            loss_type=cfg.get('loss', 'l1'),  # L1 or L2
            p2_loss_weight_gamma=cfg.get('p2_gamma', 0),
            inpaint=cfg.get('inpaint', False),
            cfg=cfg
        )

        self.boxes = BoxSampler(cfg)  # initialize box sampler

        self.prefix = prefix

        self.save_hyperparameters()

    def forward(self):
        return None

    def training_step(self, batch, batch_idx: int):
        # 获取输入和掩码
        input = batch['vol'][tio.DATA].squeeze(-1)
        orig = batch['orig'][tio.DATA].squeeze(-1)
        true_mask = batch['mask'][tio.DATA].squeeze(-1)


        # generate bboxes for DDPM
        if self.cfg.get('grid_boxes', False):  # sample boxes from a grid
            bbox = torch.zeros([input.shape[0], 4], dtype=int)
            bboxes = self.boxes.sample_grid(input)
            ind = torch.randint(0, bboxes.shape[1], (input.shape[0],))
            for j in range(input.shape[0]):
                bbox[j] = bboxes[j, ind[j]]
            bbox = bbox.unsqueeze(-1)
        else:  # sample boxes randomly
            bbox = self.boxes.sample_single_box(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else:
            noise = None

        # reconstruct
        for i in range(input.shape[0]):
            slice = input[i, 0, :, :]
            slice2 = orig[i, 0, :, :]
            true_mask_slice = true_mask[i, 0, :, :]

            array_2d = np.asarray(slice.cpu())

            # freq = get_freq_image(array_2d)
            # freq = get_freq_image_defaut(array_2d)
            freq = get_freq_multi_scale(self, array_2d)
            array_2d_mask = (true_mask_slice.cpu())
            col_sums = array_2d_mask.sum(axis=0)
            row_sums = array_2d_mask.sum(axis=1)

            nonzero_ind_col = np.nonzero(col_sums)
            nonzero_ind_row = np.nonzero(row_sums)

            total_cols = len(nonzero_ind_col)
            total_rows = len(nonzero_ind_row)

            max_patch_len = int((total_rows / 10) * 1.8)
            min_patch_len = int(max_patch_len / 2)

            # num_of_patches = random.randint(0, 5)
            num_of_patches = random.randint(0, 7)
            for k in range(num_of_patches):
                patch_height = random.randint(min_patch_len, max_patch_len)
                patch_width = random.randint(min_patch_len, max_patch_len)
                row_point = random.randint(nonzero_ind_row[0], nonzero_ind_row[-1] - 10 - max_patch_len) + 5
                col_point = random.randint(nonzero_ind_col[0], nonzero_ind_col[-1] - 10 - max_patch_len) + 5

                input[i, 0, row_point:row_point + patch_width, col_point:col_point + patch_height] = freq[
                    row_point:row_point + patch_width,
                    col_point:col_point + patch_height]
                input[i, 0, :, :] *= true_mask[i, 0, :, :].double()

        # loss, reco = self.diffusion(input, orig, box=bbox, noise=noise, mask=true_mask, healthy_weight=self.cfg['healthy_weight'], anomal_weight=self.cfg['anomal_weight'])
        loss, reco = self.diffusion(input, orig, box=bbox, noise=noise)
        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True,
                batch_size=input.shape[0], sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):

        input = batch['vol'][tio.DATA].squeeze(-1)
        orig = batch['orig'][tio.DATA].squeeze(-1)

        # generate bboxes for DDPM
        if self.cfg.get('grid_boxes', False):  # sample boxes from a grid
            bbox = torch.zeros([input.shape[0], 4], dtype=int)
            bboxes = self.boxes.sample_grid(input)
            ind = torch.randint(0, bboxes.shape[1], (input.shape[0],))
            for j in range(input.shape[0]):
                bbox[j] = bboxes[j, ind[j]]
            bbox = bbox.unsqueeze(-1)
        else:  # sample boxes randomly
            bbox = self.boxes.sample_single_box(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else:
            noise = None

        loss, reco = self.diffusion(input, orig, box=bbox, noise=noise)

        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=input.shape[0], sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160, 190, 160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self, 'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):

        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]

        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []

        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(
                4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[..., start_slice:start_slice + num_slices]
            data_orig = data_orig[..., start_slice:start_slice + num_slices]
            data_seg = data_seg[..., start_slice:start_slice + num_slices]
            data_mask = data_mask[..., start_slice:start_slice + num_slices]
            ind_offset = start_slice
        else:
            ind_offset = 0

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device=self.device)

        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3, 0, 1, 2)  # [B,C,H,W,D] -> [D,C,H,W]

        latentSpace.append(torch.tensor([0], dtype=float).repeat(input.shape[0]))  # dummy latent space

        # generate bboxes for DDPM
        bbox = self.boxes.sample_grid(input)
        reco_patched = torch.zeros_like(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else:
            noise = None

        for k in range(bbox.shape[1]):
            box = bbox[:, k]
            # reconstruct
            loss_diff, reco = self.diffusion(input, input, t=self.test_timesteps - 1, box=box, noise=noise)

            if reco.shape[1] == 2:
                reco = reco[:, 0:1, :, :]

            for j in range(reco_patched.shape[0]):
                if self.cfg.get('overlap', False):  # cut out the overlap region
                    grid = self.boxes.sample_grid_cut(input)
                    box_cut = grid[:, k]
                    if self.cfg.get('agg_overlap', 'cut') == 'cut':  # cut out the overlap region
                        reco_patched[j, :, box_cut[j, 1]:box_cut[j, 3], box_cut[j, 0]:box_cut[j, 2]] = reco[j, :,
                                                                                                       box_cut[j, 1]:
                                                                                                       box_cut[j, 3],
                                                                                                       box_cut[j, 0]:
                                                                                                       box_cut[j, 2]]
                    elif self.cfg.get('agg_overlap', 'cut') == 'avg':  # average the overlap region
                        reco_patched[j, :, box[j, 1]:box[j, 3], box[j, 0]:box[j, 2]] = reco_patched[j, :,
                                                                                       box[j, 1]:box[j, 3],
                                                                                       box[j, 0]:box[j, 2]] + reco[j, :,
                                                                                                              box[j, 1]:
                                                                                                              box[j, 3],
                                                                                                              box[j, 0]:
                                                                                                              box[j, 2]]
                else:
                    reco_patched[j, :, box[j, 1]:box[j, 3], box[j, 0]:box[j, 2]] = reco[j, :, box[j, 1]:box[j, 3],
                                                                                   box[j, 0]:box[j, 2]]

            if self.cfg.get('overlap', False) and self.cfg.get('agg_overlap',
                                                               'cut') == 'avg':  # average the intersection of all patches
                mask = torch.zeros_like(reco_patched)
                # create mask
                for k in range(bbox.shape[1]):
                    box = bbox[:, k]
                    for l in range(mask.shape[0]):
                        mask[l, :, box[l, 1]:box[l, 3], box[l, 0]:box[l, 2]] = mask[l, :, box[l, 1]:box[l, 3],
                                                                               box[l, 0]:box[l, 2]] + 1
                # divide by the mask to average the intersection of all patches
                reco_patched = reco_patched / mask

            reco = reco_patched.clone()

        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(AnomalyScoreComb)  # dummy
        AnomalyScoreReco.append(AnomalyScoreComb)  # dummy

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1, 2, 0)  # to HxWxD

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace), 0))

        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb)
        AnomalyScoreReg_vol = AnomalyScoreComb_vol  # dummy
        AnomalyScoreReco_vol = AnomalyScoreComb_vol  # dummy

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)

        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)

        # calculate metrics
        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label)

    def on_test_end(self):
        # calculate metrics
        _test_end(self)  # everything that is independent of the model choice

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    def update_prefix(self, prefix):
        self.prefix = prefix 
