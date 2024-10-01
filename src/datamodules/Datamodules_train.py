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
from src.datamodules.create_dataset import vol2slice
from torch.utils.data import ConcatDataset
log = utils.get_logger(__name__)  # init logger
from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    elem = batch[0]
    collated_batch = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated_batch[key] = default_collate([d[key] for d in batch])
        elif isinstance(elem[key], (int, float)):
            collated_batch[key] = torch.tensor([d[key] for d in batch])
        elif isinstance(elem[key], str):
            collated_batch[key] = [d[key] for d in batch]
        elif isinstance(elem[key], dict):
            # 递归调用 collate_fn 处理嵌套的字典
            collated_batch[key] = custom_collate_fn([d[key] for d in batch])
        else:
            # 对于其他类型的数据，直接组成列表
            collated_batch[key] = [d[key] for d in batch]
    return collated_batch
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
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False), collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)

class CombinedDataModule(LightningDataModule):
    def __init__(self, cfg, fold=None):
        super(CombinedDataModule, self).__init__()
        self.cfg = cfg
        self.fold = fold
        self.ixi_module = IXI(cfg, fold)
        self.cached_brats_train_path = os.path.join(cfg.path.cache_dir, 'cached_brats_train.pkl')
        self.cached_brats_val_path = os.path.join(cfg.path.cache_dir, 'cached_brats_val.pkl')

    def setup(self, stage: Optional[str] = None):
        # 处理 IXI 数据集
        self.ixi_module.setup(stage)
        log.info("IXI dataset setup complete.")

        # 设置缓存目录
        train_cache_dir = os.path.join(self.cfg.path.cache_dir, 'brats_train')
        val_cache_dir = os.path.join(self.cfg.path.cache_dir, 'brats_val')

        # 加载或处理 BraTS21 数据集，获取 subject 文件路径列表
        self.brats_train_subject_paths = self.load_or_process_brats(self.cfg.path.BraTS21.IDs.train, train_cache_dir)
        self.brats_val_subject_paths = self.load_or_process_brats(self.cfg.path.BraTS21.IDs.val, val_cache_dir)

        # 创建 LazyBraTSDataset 实例
        brats_train_dataset = LazyBraTSDataset(self.brats_train_subject_paths, transform=get_transform(self.cfg))
        brats_val_dataset = LazyBraTSDataset(self.brats_val_subject_paths, transform=get_transform(self.cfg))
        brats_train_dataset = vol2slice(brats_train_dataset, self.cfg)
        brats_val_dataset = vol2slice(brats_val_dataset, self.cfg)

        print(f"Type of self.ixi_module.train: {type(self.ixi_module.train)}")
        print(f"Type of self.ixi_module.val: {type(self.ixi_module.val)}")
        # 合并 IXI 数据集和 BraTS21 数据集
        combined_train_dataset = ConcatDataset([self.ixi_module.train, brats_train_dataset])
        combined_val_dataset = ConcatDataset([self.ixi_module.val, brats_val_dataset])
        self.train = combined_train_dataset
        self.val = combined_val_dataset
        log.info(f"Combined dataset setup completed. Train dataset size: {len(self.train)}, Validation dataset size: {len(self.val)}")
    def train_dataloader(self) -> None:
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                                pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False), collate_fn=custom_collate_fn)
    def val_dataloader(self) -> None:
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                              pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)
    # 这里需要改成两个数据集合并的验证集
    def val_eval_dataloader(self):
        return DataLoader(self.ixi_module.val_eval, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)
    def test_eval_dataloader(self):
        return DataLoader(self.ixi_module.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)
        

    def load_or_process_brats(self, csv_path, cache_dir):
        index_file_path = os.path.join(cache_dir, 'subjects_index.pkl')
        if os.path.exists(index_file_path):
            return self.load_cached_subjects(cache_dir)
        else:
            return self.process_and_cache_brats(csv_path, cache_dir)

    def process_and_cache_brats(self, csv_path, cache_dir, batch_size=100):
        df = pd.read_csv(csv_path)
        df['img_path'] = df['img_path'].apply(lambda x: os.path.join(self.cfg.path.pathBase + '/Data/', x.lstrip('/')))
        df['mask_path'] = df['mask_path'].apply(lambda x: os.path.join(self.cfg.path.pathBase + '/Data/', x.lstrip('/')))
        df['seg_path'] = df['seg_path'].apply(lambda x: os.path.join(self.cfg.path.pathBase + '/Data/', x.lstrip('/')))

        os.makedirs(cache_dir, exist_ok=True)
        index_file_path = os.path.join(cache_dir, 'subjects_index.pkl')
        subject_paths = []

        num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            log.info(f"Processing batch {batch_num + 1}/{num_batches}...")

            brats_subjects = self.process_brats_batch(batch_df)

            for subject in brats_subjects:
                subject_id = subject['ID']
                subject_cache_path = os.path.join(cache_dir, f"{subject_id}.pkl")
                with open(subject_cache_path, 'wb') as f:
                    pickle.dump(subject, f)
                subject_paths.append(subject_cache_path)

            # 释放内存
            del brats_subjects
            gc.collect()

        # 保存索引文件
        with open(index_file_path, 'wb') as f:
            pickle.dump(subject_paths, f)

        return subject_paths  # 返回 subject 文件路径列表
    def load_cached_subjects(self, cache_dir):
        index_file_path = os.path.join(cache_dir, 'subjects_index.pkl')
        if not os.path.exists(index_file_path):
            log.error(f"Index file not found at {index_file_path}")
            raise FileNotFoundError(f"Index file not found at {index_file_path}")
        with open(index_file_path, 'rb') as f:
            subject_paths = pickle.load(f)
        log.info(f"Loaded {len(subject_paths)} subject paths from index.")
        return subject_paths

    def process_brats_batch(self, batch_df):
        brats_subjects = []
        processed_ids = set()

        patient_slices = {}

        for _, sub in batch_df.iterrows():
            if not os.path.exists(sub.img_path):
                log.warning(f"Image path does not exist: {sub.img_path}")
                continue

            try:
                vol_img = tio.ScalarImage(sub.img_path, reader=sitk_reader)
                seg_img = tio.LabelMap(sub.seg_path, reader=sitk_reader) if sub.seg_path else None
                mask_img = tio.LabelMap(sub.mask_path, reader=sitk_reader) if sub.mask_path else None

                if vol_img.data is None or (seg_img and seg_img.data is None):
                    log.warning(f"Failed to load image or segmentation for: {sub.img_path}")
                    continue

                non_tumor_slices = []

                for slice_index in range(seg_img.data.shape[-1]):
                    unique_id = f"{sub.img_name}_slice_{slice_index}"
                    if unique_id in processed_ids:
                        log.warning(f"Duplicate slice detected: {unique_id}, skipping...")
                        continue

                    if not seg_img.data[..., slice_index].any():
                        non_tumor_slices.append({
                            'vol': vol_img.data[..., slice_index],
                            'mask': mask_img.data[..., slice_index] if mask_img else None,
                            'slice_index': slice_index
                        })
                        processed_ids.add(unique_id)

                if sub.img_name not in patient_slices:
                    patient_slices[sub.img_name] = {'vol': [], 'mask': []}

                for slice_data in non_tumor_slices:
                    patient_slices[sub.img_name]['vol'].append(slice_data['vol'])
                    if slice_data['mask'] is not None:
                        patient_slices[sub.img_name]['mask'].append(slice_data['mask'])

            except Exception as e:
                log.error(f"Error processing subject {sub.img_name}: {e}")
                continue

        for patient_id, slices in patient_slices.items():
            vol_3d = torch.stack(slices['vol'], dim=-1).float()
            mask_3d = torch.stack(slices['mask'], dim=-1) if slices['mask'] else None

            subject_dict = {
                'orig': tio.ScalarImage(tensor=vol_3d),
                'vol': tio.ScalarImage(tensor=vol_3d.clone()),
                'age': sub.age,
                'ID': patient_id,  # 确保 ID 唯一
                'label': sub.label,
                'Dataset': sub.setname,
                'stage': sub.settype,
                'path': sub.img_path
            }

            if mask_3d is not None:
                subject_dict['mask'] = tio.LabelMap(tensor=mask_3d)

            subject = tio.Subject(subject_dict)
            brats_subjects.append(subject)

        return brats_subjects


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
def get_transform(cfg):
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
                tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline',
                             exclude=exclude_from_resampling),
            ])
        else:
            preprocess = tio.Compose([
                tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                     masking_method='mask'),
                tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline',
                             exclude=exclude_from_resampling),
            ])
        return preprocess
    except Exception as e:
        log.error(f"Error in get_transform: {e}")
        raise

class LazyBraTSDataset(Dataset):
    def __init__(self, subject_paths, transform=None):
        self.subject_paths = subject_paths
        self.transform = transform

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, index):
        subject_path = self.subject_paths[index]
        with open(subject_path, 'rb') as f:
            subject = pickle.load(f)
        if self.transform:
            subject = self.transform(subject)
        return subject