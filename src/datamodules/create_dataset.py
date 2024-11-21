from torch.utils.data import Dataset
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
from PIL import Image
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager
import random
from src.utils import utils
log = utils.get_logger(__name__)

def Train(csv, cfg, preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        for i in range(6):

            subject_dict = {
                'orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
                'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
                'age': sub.age,
                'ID': sub.img_name,
                'label': sub.label,
                'Dataset': sub.setname,
                'stage': sub.settype,
                'path': sub.img_path
            }
            if sub.mask_path != None:  # if we have masks
                subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
            else:  # if we don't have masks, we create a mask from the image
                subject_dict['mask'] = tio.LabelMap(tensor=tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0)

            subject = tio.Subject(subject_dict)
            subjects.append(subject)


    if preload:
        manager = Manager()
        cache = DatasetCache(manager)
        ###subject: Image
        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment=get_augment(cfg))
    else:
        ds = tio.SubjectsDataset(subjects, transform=tio.Compose([get_transform(cfg), get_augment(cfg)]))

    if cfg.spatialDims == '2D':
        slice_ind = cfg.get('startslice', None)
        seq_slices = cfg.get('sequentialslices', None)
        ds = vol2slice(ds, cfg, slice=slice_ind, seq_slices=seq_slices)

    return ds


def Eval(csv, cfg):
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path, reader=sitk_reader).shape != tio.ScalarImage(
                sub.mask_path, reader=sitk_reader).shape:
            print(
                f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path, reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path, reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')

        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'vol_orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            # we need the image in original size for evaluation
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'seg_available': False,
            'path': sub.img_path}
        if sub.seg_path != None:  # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path, reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,
                                                    reader=sitk_reader)  # we need the image in original size for evaluation
            subject_dict['seg_available'] = True
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
            subject_dict['mask_orig'] = tio.LabelMap(sub.mask_path,
                                                     reader=sitk_reader)  # we need the image in original size for evaluation
        else:
            tens = tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0
            subject_dict['mask'] = tio.LabelMap(tensor=tens)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=tens)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
    return ds


## got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)


class preload_wrapper(Dataset):
    def __init__(self, ds, cache, augment=None):
        self.cache = cache
        self.ds = ds
        self.augment = augment
        log.info(f"preload initialized with self.ds type: {type(self.ds)}")

    def reset_memory(self):
        self.cache.reset()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if self.cache.is_cached(index):
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject
    @property
    def subjects(self):
        try:
            return self.ds._subjects  # 直接访问受保护的 _subjects 属性
        except AttributeError:
            log.error(f"'_subjects' attribute not found in {type(self.ds)}")
            raise


class vol2slice(Dataset):
    def __init__(self, ds, cfg, onlyBrain=False, slice=None, seq_slices=None, only_tumor=False):
        self.ds = ds
        self.onlyBrain = onlyBrain
        self.slice = slice
        self.seq_slices = seq_slices
        self.only_tumor = only_tumor
        self.counter = 0
        self.ind = None
        self.cfg = cfg
        log.info(f"vol2slice initialized with self.ds type: {type(self.ds)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)

        if 'orig' not in subject:
            log.error(f"'orig' key missing in subject at index {index}")
            log.error(f"Subject keys: {list(subject.keys())}")
            raise KeyError("'orig' key missing in subject")

        # Determine valid slice indices based on brain mask
        if self.onlyBrain:
            start_ind = None
            stop_ind = None
            for i in range(subject['vol'].data.shape[-1]):
                mask_slice = subject['mask'].data[0, :, :, i]
                if mask_slice.any():
                    if start_ind is None:
                        start_ind = i
                    stop_ind = i  # Update stop_ind to the last valid slice
            if start_ind is None or stop_ind is None:
                log.warning(f"No brain slices found in subject at index {index}")
                return self.__getitem__((index + 1) % len(self.ds))
            low = start_ind
            high = stop_ind + 1  # Include the last valid slice
        else:
            low = 0
            high = subject['vol'].data.shape[-1]

        # Select slice index
        if self.slice is not None:
            ind = self.slice
            if self.seq_slices is not None:
                low = ind
                high = ind + self.seq_slices
                ind = torch.randint(low, high, size=(1,)).item()
        else:
            if self.cfg.get('unique_slice', False):
                if self.counter % self.cfg.batch_size == 0:
                    ind = torch.randint(low, high, size=(1,)).item()
                    self.last_ind = ind
                else:
                    ind = self.last_ind
                self.counter += 1
            else:
                ind = torch.randint(low, high, size=(1,)).item()

        ind_tensor = torch.tensor([ind])

        # Check if the slice contains tumor (for datasets with segmentation)
        contains_tumor = False
        if 'seg' in subject:
            seg_slice = subject['seg'].data[..., ind]
            contains_tumor = seg_slice.any()
        else:
            # If no 'seg' field, assume it's a healthy subject
            contains_tumor = False

        # Apply 'only_tumor' filter
        if self.only_tumor:
            if not contains_tumor:
                # Current slice does not contain tumor, try next
                return self.__getitem__((index + 1) % len(self.ds))
        else:
            if contains_tumor:
                # Current slice contains tumor, but we only want healthy slices
                return self.__getitem__((index + 1) % len(self.ds))

        # Update subject's data to only include the selected slice
        subject['ind'] = ind
        subject['vol'].data = subject['vol'].data[..., ind_tensor]
        subject['orig'].data = subject['orig'].data[..., ind_tensor]
        if 'mask' in subject:
            subject['mask'].data = subject['mask'].data[..., ind_tensor]
        if 'seg' in subject:
            subject['seg'].data = subject['seg'].data[..., ind_tensor]

        # Check if the mask is valid (not all zeros)
        if 'mask' in subject:
            if not subject['mask'].data.any() or subject['mask'].data.sum() < 600:
                # Mask is invalid, try next slice
                return self.__getitem__((index + 1) % len(self.ds))

        return subject

    @property
    def subjects(self):
        try:
            return self.ds.subjects  # Try accessing subjects attribute directly
        except AttributeError:
            try:
                return self.ds._subjects  # Fallback to _subjects attribute
            except AttributeError:
                log.error(f"'subjects' or '_subjects' attribute not found in {type(self.ds)}")
                raise


def get_transform(cfg):  # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim', (160, 192, 160)))

    if not cfg.resizedEvaluation:
        exclude_from_resampling = ['vol_orig', 'mask_orig', 'seg_orig']
    else:
        exclude_from_resampling = None

    if cfg.get('unisotropic_sampling', True):
        preprocess = tio.Compose([
            tio.CropOrPad((h, w, d), padding_mode=0),
            tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                 masking_method='mask'),
            tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            # ,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else:
        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                 masking_method='mask'),
            tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            # ,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    return preprocess


def get_augment(cfg):  # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    augment = tio.Compose(augmentations)
    return augment


def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path):  # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
    return vol, None


