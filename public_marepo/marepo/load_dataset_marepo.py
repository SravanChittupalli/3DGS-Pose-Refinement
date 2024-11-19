from torch.utils.data import DataLoader
from torch.utils.data import sampler
from dataset import CamLocDataset
from marepo.dataset_marepo import CamLocDatasetAll
import torch
import numpy as np
import random

def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_single_scene(options):
    ''' A wrapper function to load single scene dataset'''
    # Create dataset.
    num_data_loader_workers = 8

    # Create train mapping dataset.
    train_map_dataset = CamLocDataset(
        root_dir=options.scene / "train",
        mode=0,  # Default for marepo, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        augment=options.use_aug,
        aug_rotation=options.aug_rotation,
        aug_scale_max=options.aug_scale,
        aug_scale_min=1 / options.aug_scale,
        num_clusters=options.num_clusters,  # Optional clustering for Cambridge experiments.
        cluster_idx=options.cluster_idx,  # Optional clustering for Cambridge experiments.
        load_rgb=options.load_rgb # load image in 3 rgb channels instead of 1 gray channel
    )

    train_dl = DataLoader(
        train_map_dataset,
        shuffle=True,
        batch_size=options.batch_size,
        num_workers=num_data_loader_workers)

    # Create train query dataset.
    train_query_dataset = CamLocDataset(
        root_dir=options.scene / "test",
        mode=0,  # Default for marepo, we don't need scene coordinates/RGB-D.
        use_half=options.use_val_half, # Default is True for ACE, False for posenet
        image_height=options.image_resolution,
        load_rgb=options.load_rgb
    )

    val_dl = DataLoader(
        train_query_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_data_loader_workers)

    test_dl = val_dl
    return train_dl, val_dl, test_dl

def load_single_map_scene(options, scene_dir):
    ''' A wrapper function to load single scene dataset'''
    base_seed = 2089
    set_seed(base_seed)

    # Used to generate batch indices.
    batch_generator = torch.Generator()
    batch_generator.manual_seed(base_seed + 1023)

    # Dataloader generator, used to seed individual workers by the dataloader.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(base_seed + 511)

    num_data_loader_workers = 4

    print("load map dataset: ", scene_dir / "train")

    # Create train mapping dataset.
    map_dataset = CamLocDataset(
        root_dir=scene_dir / "train", # mapping dataset of the specific scene
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        augment=options.use_aug,
        aug_rotation=options.aug_rotation,
        aug_scale_max=options.aug_scale,
        aug_scale_min=1 / options.aug_scale,
        num_clusters=options.num_clusters,  # Optional clustering for Cambridge experiments.
        cluster_idx=options.cluster_idx,  # Optional clustering for Cambridge experiments.
    )

    # Sampler.
    batch_sampler = sampler.BatchSampler(sampler.RandomSampler(map_dataset, generator=batch_generator),
                                         batch_size=1,
                                         drop_last=False)

    # Used to seed workers in a reproducible manner.
    def seed_worker(worker_id):
        # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
        # the dataloader generator.
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
    # need to rescale all images in the batch to the same size).
    map_dl = DataLoader(dataset=map_dataset,
                        sampler=batch_sampler,
                        batch_size=None,
                        worker_init_fn=seed_worker,
                        generator=loader_generator,
                        pin_memory=True,
                        num_workers=num_data_loader_workers,
                        persistent_workers=num_data_loader_workers > 0,
                        timeout=20 if num_data_loader_workers > 0 else 0,
                        )

    return map_dl, map_dataset

def load_multiple_scene(options):
    ''' A wrapper function to load all scenes in the dataset'''
    num_data_loader_workers = 12

    non_mapfree_dataset_naming=False
    # load "map", "query", "map+query"
    if options.preprocessing:
        # we don't need to shuffle at preprocessing stage
        train_shuffle = False
        train_split = ["map", "query"]
        val_split = ["map", "query"]
        test_split = ["map", "query"]
        train_batch_size=1
        val_batch_size=1
        test_batch_size=1
        load_sc_map=False
        load_scheme2_sc_map=False
        train_augment=True if options.scheme3 else False
        load_scheme3_sc_map = False
        marepo_sc_augment = False
        jitter_trans = 0.0
        jitter_rot = 0.0
        load_mapping_buffer_features=False
        random_mapping_buffers=False
        all_mapping_buffers=False
        center_crop=options.center_crop
    else:
        train_shuffle = True
        if options.finetune==True: # finetune the mapping set from the test scenes
            train_split = ["map"]
            val_split = ["query"]
            test_split = ["query"]
        else:
            if options.train_mapping_query:
                train_split=["map", "query"]
            else:
                train_split = ["query"]
            val_split = ["query"]
            test_split = ["query"]
        train_batch_size = options.batch_size
        val_batch_size = options.val_batch_size
        test_batch_size = options.test_batch_size
        load_sc_map=True
        load_scheme2_sc_map=options.load_scheme2_sc_map
        load_scheme3_sc_map = options.load_scheme3_sc_map
        train_augment=False
        marepo_sc_augment=options.marepo_sc_augment # if True, we apply additional data augmentation for training Marepo
        if marepo_sc_augment==True:
            non_mapfree_dataset_naming=options.non_mapfree_dataset_naming
        jitter_trans = options.jitter_trans
        jitter_rot = options.jitter_rot
        load_mapping_buffer_features = options.fuse_mapping_confidence
        random_mapping_buffers = options.random_mapping_buffers
        all_mapping_buffers = options.all_mapping_buffers
        center_crop=options.center_crop

    train_dataset = CamLocDatasetAll(
        root_dir=options.dataset_path/"train",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        split=train_split,
        trainskip=options.trainskip,
        load_sc_map=load_sc_map,
        load_scheme2_sc_map=load_scheme2_sc_map,
        augment=train_augment, # use data augmentation if True
        load_scheme3_sc_map=load_scheme3_sc_map,
        marepo_sc_augment=marepo_sc_augment,
        jitter_trans=jitter_trans,
        jitter_rot=jitter_rot,
        load_mapping_buffer_features=load_mapping_buffer_features,
        random_select_mapping_buffer=random_mapping_buffers,
        all_mapping_buffers=all_mapping_buffers,
        non_mapfree_dataset_naming=non_mapfree_dataset_naming,
        center_crop=center_crop
    )

    train_dl = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        batch_size=train_batch_size,
        num_workers=num_data_loader_workers)

    val_dataset = CamLocDatasetAll(
        root_dir=options.dataset_path / "val",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        split=val_split,
        trainskip=options.testskip,
        load_sc_map=load_sc_map,
        load_scheme2_sc_map=load_scheme2_sc_map,
        load_mapping_buffer_features=load_mapping_buffer_features,
        all_mapping_buffers=all_mapping_buffers,
        center_crop=center_crop
    )

    val_dl = DataLoader(
        val_dataset,
        shuffle=False, # True just for experiment purpose
        batch_size=val_batch_size,
        num_workers=num_data_loader_workers
    )

    test_dataset = CamLocDatasetAll(
        root_dir=options.dataset_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        split=test_split,
        trainskip=options.testskip,
        load_sc_map=load_sc_map,
        load_scheme2_sc_map=load_scheme2_sc_map,
        load_mapping_buffer_features=load_mapping_buffer_features,
        all_mapping_buffers=all_mapping_buffers,
        center_crop=center_crop
    )

    test_dl = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=test_batch_size,
        num_workers=num_data_loader_workers)
    
    return train_dl, val_dl, test_dl


def load_multiple_map_only_scene(options):
    ''' A wrapper function to load all mapping scenes in the dataset
    It is designed for generating ACE like mapping feature buffer
    '''
    num_data_loader_workers = 8

    # load "map", "query", "map+query"
    assert(options.preprocessing==True)
    # we don't need to shuffle at preprocessing stage
    train_shuffle = False
    train_split = ["map"]
    val_split = ["map"]
    test_split = ["map"]
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    load_sc_map = False
    load_scheme2_sc_map = False
    train_augment = True # assume we need data augmentation
    load_scheme3_sc_map = False
    marepo_sc_augment = False

    train_dataset = CamLocDatasetAll(
        root_dir=options.dataset_path / "train",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        split=train_split,
        trainskip=options.trainskip,
        load_sc_map=load_sc_map,
        load_scheme2_sc_map=load_scheme2_sc_map,
        augment=train_augment,
        load_scheme3_sc_map=load_scheme3_sc_map,
        marepo_sc_augment=marepo_sc_augment,
    )

    train_dl = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        batch_size=train_batch_size,
        num_workers=num_data_loader_workers)

    val_dataset = CamLocDatasetAll(
        root_dir=options.dataset_path / "val",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        split=val_split,
        trainskip=options.testskip,
        load_sc_map=load_sc_map,
        load_scheme2_sc_map=load_scheme2_sc_map,
        augment=train_augment
    )

    val_dl = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=num_data_loader_workers
    )

    test_dataset = CamLocDatasetAll(
        root_dir=options.dataset_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=options.use_half,
        image_height=options.image_resolution,
        split=test_split,
        trainskip=options.testskip,
        load_sc_map=load_sc_map,
        load_scheme2_sc_map=load_scheme2_sc_map,
        augment=train_augment
    )

    test_dl = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=test_batch_size,
        num_workers=num_data_loader_workers)

    return train_dl, val_dl, test_dl

