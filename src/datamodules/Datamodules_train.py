from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd
import pickle
import os
import torchio as tio
import torch
import re
from src.utils import utils
import SimpleITK as sitk
from torch.utils.data import Dataset
import gc  
import tracemalloc  
log = utils.get_logger(__name__)  # init logger

class IXI(LightningDataModule):
    def __init__(self, cfg, fold=None):
        super(IXI, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False  # no permutation for IXI

        self.csv = {}
        states = ['train', 'val', 'test']

        try:
            self.csv['train'] = pd.read_csv(cfg.path.IXI.IDs.train[fold])
            self.csv['val'] = pd.read_csv(cfg.path.IXI.IDs.val[fold])
            self.csv['test'] = pd.read_csv(cfg.path.IXI.IDs.test)
        except Exception as e:
            log.error(f"Error loading CSV files for IXI dataset: {e}")
            raise

        if cfg.mode == 't2':
            try:
                keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2)  # only keep t2 images that have a t1 counterpart
            except Exception as e:
                log.error(f"Error loading keep_t2 CSV: {e}")
                raise

        for state in states:
            try:
                self.csv[state]['settype'] = state
                self.csv[state]['setname'] = 'IXI'
                self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
                self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
                self.csv[state]['seg_path'] = None

                if cfg.mode == 't2':
                    self.csv[state] = self.csv[state][self.csv[state].img_name.isin(keep_t2['0'].str.replace('t2', 't1'))]
                    self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1', 't2')
            except Exception as e:
                log.error(f"Error processing {state} data in IXI dataset: {e}")
                raise

    def setup(self, stage: Optional[str] = None):
        try:
            if not hasattr(self, 'train'):
                if self.cfg.sample_set:  # 用于调试
                    self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                    self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                    self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                    self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
                else:
                    self.train = create_dataset.Train(self.csv['train'], self.cfg)
                    self.val = create_dataset.Train(self.csv['val'], self.cfg)
                    self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                    self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

                log.info(f"IXI setup completed. Train subjects: {len(self.train.ds)}, Validation subjects: {len(self.val.ds)}")
        except Exception as e:
            log.error(f"Error during IXI setup: {e}")
            raise

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

class CombinedDataModule(LightningDataModule):
    def __init__(self, cfg, fold=None):
        super(CombinedDataModule, self).__init__()
        self.cfg = cfg
        self.fold = fold
        self.ixi_module = IXI(cfg, fold)
        self.cached_brats_train_path = os.path.join(cfg.path.cache_dir, 'cached_brats_train.pkl')
        self.cached_brats_val_path = os.path.join(cfg.path.cache_dir, 'cached_brats_val.pkl')

    def setup(self, stage: Optional[str] = None):
        # 处理IXI数据集
        self.ixi_module.setup(stage)
        log.info("IXI dataset setup complete.")

        # 处理BraTS21数据集，如果已经存在缓存则直接加载，否则进行处理
        self.brats_train_subjects = list(self.load_or_process_brats(self.cfg.path.BraTS21.IDs.train, self.cached_brats_train_path))
        self.brats_val_subjects = list(self.load_or_process_brats(self.cfg.path.BraTS21.IDs.val, self.cached_brats_val_path))

        # 确保数据集加载后不为空
        if not self.brats_train_subjects:
            raise ValueError("brats_train_subjects is empty after processing.")
        if not self.brats_val_subjects:
            raise ValueError("brats_val_subjects is empty after processing.")

        # 合并IXI和BraTS21数据集
        combined_train_subjects = self.ixi_module.train.ds.subjects + self.brats_train_subjects
        combined_val_subjects = self.ixi_module.val.ds.subjects + self.brats_val_subjects

        # 这里使用vol2slice仅用于切片选择
        self.train = vol2slice(tio.SubjectsDataset(combined_train_subjects, transform=get_transform(self.cfg)), self.cfg)
        self.val = vol2slice(tio.SubjectsDataset(combined_val_subjects, transform=get_transform(self.cfg)), self.cfg)

        log.info(f"Combined dataset setup completed. Train subjects: {len(self.train)}, Validation subjects: {len(self.val)}")

    def load_or_process_brats(self, csv_path, cache_path):
        # 检查是否存在缓存文件
        if os.path.exists(cache_path):
            return self.load_cached_subjects(cache_path)
        else:
            # 如果不存在缓存，则处理BraTS21数据并缓存结果
            return self.process_and_cache_brats(csv_path, cache_path)

    def process_and_cache_brats(self, csv_path, cache_path, batch_size=100):
        df = pd.read_csv(csv_path)
        df['img_path'] = df['img_path'].apply(lambda x: os.path.join(self.cfg.path.pathBase + '/Data/', x.lstrip('/')))
        df['mask_path'] = df['mask_path'].apply(lambda x: os.path.join(self.cfg.path.pathBase + '/Data/', x.lstrip('/')))
        df['seg_path'] = df['seg_path'].apply(lambda x: os.path.join(self.cfg.path.pathBase + '/Data/', x.lstrip('/')))

        num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            log.info(f"Processing batch {batch_num + 1}/{num_batches}...")

            brats_subjects = self.process_brats_batch(batch_df)

            # 每个批次处理完成后直接缓存
            with open(cache_path, 'ab') as f:
                pickle.dump(brats_subjects, f)

            del brats_subjects  # 释放内存
            gc.collect()  # 手动触发垃圾回收

        # 重新加载缓存的数据
        return self.load_cached_subjects(cache_path)

    def load_cached_subjects(self, cache_path):
        loaded_count = 0
        try:
            with open(cache_path, 'rb') as f:
                while True:
                    try:
                        batch_subjects = pickle.load(f)
                        loaded_count += len(batch_subjects)
                        yield from batch_subjects
                    except EOFError:
                        break
            log.info(f"Loaded {loaded_count} subjects from cache.")
        except Exception as e:
            log.error(f"Error loading cached BraTS21 subjects: {e}")
            raise

    def process_brats_batch(self, batch_df):
        brats_subjects = []
        processed_ids = set()  # 用于存储已处理的ID
        for _, sub in batch_df.iterrows():
            if not os.path.exists(sub.img_path):
                log.warning(f"Image path does not exist: {sub.img_path}")
                continue

            try:
                vol_img = tio.ScalarImage(sub.img_path, reader=sitk_reader)
                seg_img = tio.LabelMap(sub.seg_path, reader=sitk_reader) if sub.seg_path is not None else None
                mask_img = tio.LabelMap(sub.mask_path, reader=sitk_reader) if sub.mask_path is not None else None

                if vol_img.data is None or (seg_img and seg_img.data is None):
                    log.warning(f"Failed to load image or segmentation for: {sub.img_path}")
                    continue

                non_tumor_slices = []

                for slice_index in range(seg_img.data.shape[-1]):  # 逐切片检查
                    unique_id = f"{sub.img_name}_slice_{slice_index}"
                    if unique_id in processed_ids:
                        log.warning(f"Duplicate slice detected: {unique_id}, skipping...")
                        continue

                    if not seg_img.data[..., slice_index].any():  # 如果该切片没有非零值
                        non_tumor_slices.append({
                            'vol': vol_img.data[..., slice_index],  # 对应的影像切片
                            'seg': seg_img.data[..., slice_index] if seg_img else None,  # 对应的分割标签切片
                            'mask': mask_img.data[..., slice_index] if mask_img else None,  # 对应的脑部掩码切片
                            'slice_index': slice_index
                        })
                        processed_ids.add(unique_id)

                if non_tumor_slices:
                    for slice_data in non_tumor_slices:
                        slice_vol = slice_data['vol'].unsqueeze(-1)  # 恢复到3D形状（H, W, 1）
                        slice_seg = slice_data['seg'].unsqueeze(-1) if slice_data['seg'] is not None else None
                        slice_mask = slice_data['mask'].unsqueeze(-1) if slice_data['mask'] is not None else None

                        subject_dict = {
                            'orig': tio.ScalarImage(tensor=slice_vol),
                            'vol': tio.ScalarImage(tensor=slice_vol),
                            'age': sub.age,
                            'ID': f"{sub.img_name}_slice_{slice_data['slice_index']}",
                            'label': sub.label,
                            'Dataset': sub.setname,
                            'stage': sub.settype,
                            'path': sub.img_path
                        }

                        if slice_seg is not None:
                            subject_dict['seg'] = tio.LabelMap(tensor=slice_seg)
                        if slice_mask is not None:
                            subject_dict['mask'] = tio.LabelMap(tensor=slice_mask)
                        # 去重检查，避免重复处理相同的切片
                        subject = tio.Subject(subject_dict)
                        if subject.ID not in [s.ID for s in brats_subjects]:
                            brats_subjects.append(subject)
                            log.info(f"Successfully loaded and processed slice {slice_index} of subject: {sub.img_name}")
                        else:
                            log.warning(f"Duplicate slice detected: {subject.ID}, skipping...")

            except Exception as e:
                log.error(f"Error processing subject {sub.img_name}: {e}")
                continue

        return brats_subjects

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return self.ixi_module.test_eval_dataloader()
def sitk_reader(path):
    try:
        image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
        vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
        return vol, None
    except Exception as e:
        log.error(f"Error reading image at {path}: {e}")
    raise
def get_transform(cfg):  # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim', (160, 192, 160)))
    if not cfg.resizedEvaluation:
        exclude_from_resampling = ['vol_orig', 'mask_orig', 'seg_orig']
    else:
        exclude_from_resampling = None

    try:
        if cfg.get('unisotropic_sampling', True):
            preprocess = tio.Compose([
                tio.CropOrPad((h, w, d), padding_mode=0),
                tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                    masking_method='mask'),
                tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            ])
        else:
            preprocess = tio.Compose([
                tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                    masking_method='mask'),
                tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            ])
        return preprocess
    except Exception as e:
        log.error(f"Error in get_transform: {e}")
        raise
class vol2slice(Dataset):
    def __init__(self, ds, cfg, onlyBrain=False, slice=None, seq_slices=None):
        self.ds = ds
        self.onlyBrain = onlyBrain
        self.slice = slice
        self.seq_slices = seq_slices
        self.counter = 0
        self.ind = None
        self.cfg = cfg
        if not isinstance(self.ds, tio.SubjectsDataset):
            log.error("vol2slice initialized with a non-SubjectsDataset object.")
            raise TypeError("The provided ds is not an instance of tio.SubjectsDataset.")        
        log.info(f"vol2slice initialized with {len(self.ds.subjects)} subjects.")
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        try:
            subject = self.ds.__getitem__(index)
            if self.onlyBrain:
                start_ind = None
                for i in range(subject['vol'].data.shape[-1]):
                    if subject['mask'].data[0, :, :, i].any() and start_ind is None:
                        start_ind = i
                    if not subject['mask'].data[0, :, :, i].any() and start_ind is not None:
                        stop_ind = i
                low = start_ind
                high = stop_ind
            else:
                low = 0
                high = subject['vol'].data.shape[-1]
            if self.slice is not None:
                self.ind = self.slice
                if self.seq_slices is not None:
                    low = self.ind
                    high = self.ind + self.seq_slices
                    self.ind = torch.randint(low, high, size=[1])
            else:
                if self.cfg.get('unique_slice', False):
                    if self.counter % self.cfg.batch_size == 0 or self.ind is None:
                        self.ind = torch.randint(low, high, size=[1])
                    self.counter += 1
                else:
                    self.ind = torch.randint(low, high, size=[1])

            subject['ind'] = self.ind
            subject['vol'].data = subject['vol'].data[..., self.ind]
            subject['mask'].data = subject['mask'].data[..., self.ind]
            subject['orig'].data = subject['orig'].data[..., self.ind]

            return subject
        except Exception as e:
            log.error(f"Error in vol2slice __getitem__ for index {index}: {e}")
            raise