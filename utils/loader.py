import torch
from timm.data.transforms_factory import create_transform
from timm.data.loader import (fast_collate,
                              MultiEpochsDataLoader,
                              PrefetchLoader)
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.constants import (IMAGENET_DEFAULT_MEAN,
                                 IMAGENET_DEFAULT_STD)
from .sampler import RepeatAugmentSampler

def create_loader(dataset,
                  input_size,
                  batch_size,
                  is_training=False,
                  use_prefetcher=True,
                  no_aug=False,
                  re_prob=0.,
                  re_mode='const',
                  re_count=1,
                  re_split=False,
                  scale=None,
                  ratio=None,
                  hflip=0.5,
                  vflip=0.,
                  color_jitter=0.4,
                  auto_augment=None,
                  num_aug_splits=0,
                  interpolation='bilinear',
                  mean=IMAGENET_DEFAULT_MEAN,
                  std=IMAGENET_DEFAULT_STD,
                  num_workers=1,
                  distributed=False,
                  crop_pct=None,
                  collate_fn=None,
                  pin_memory=False,
                  fp16=False,
                  tf_preprocessing=False,
                  use_multi_epochs_loader=False,
                  repeated_aug=False):
    re_num_splits = 0
    if re_split:
        re_num_splits = num_aug_splits or 2
    
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    
    sampler = None
    
    if distributed:
        if is_training:
            if repeated_aug:
                sampler = RepeatAugmentSampler(dataset)
                print('=use repeated augmentation=')
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)
    
    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate
    
    loader_class = torch.utils.data.DataLoader
    
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    
    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
    )
    
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )
    
    return loader
