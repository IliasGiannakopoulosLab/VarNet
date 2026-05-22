import os
import torch
import numpy as np
import h5py
import logging
import pickle
import random
import pytorch_lightning as pl
import torch.distributed as dist
import xml.etree.ElementTree as etree

from torch.utils.data import Sampler
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

# -------------------------------------------#
# -------- worker seed initialization ------ #
# -------------------------------------------#

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    base_seed = worker_info.seed

    is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    if isinstance(dataset, (CombinedSliceDataset, CombinedVolumeDataset)):
        for i, ds in enumerate(dataset.datasets):
            if ds.transform.mask_func is None:
                continue

            if is_ddp:
                seed = (
                    base_seed
                    - worker_info.id
                    + torch.distributed.get_rank()
                    * (worker_info.num_workers * len(dataset.datasets))
                    + worker_info.id * len(dataset.datasets)
                    + i
                )
            else:
                seed = base_seed - worker_info.id + worker_info.id * len(dataset.datasets) + i

            ds.transform.mask_func.rng.seed(seed % (2**32 - 1))
    else:
        if dataset.transform.mask_func is None:
            return

        if is_ddp:
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed

        dataset.transform.mask_func.rng.seed(seed % (2**32 - 1))


# -------------------------------------------#
# -------- utility: not both set ----------- #
# -------------------------------------------#
def _check_both_not_none(v1, v2):
    return (v1 is not None) and (v2 is not None)


# -------------------------------------------#
# -------- xml query helper ---------------- #
# -------------------------------------------#
def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:

    prefix = "ns"
    ns = {prefix: namespace}
    query = "."

    for el in qlist:
        query += f"//{prefix}:{el}"

    value = root.find(query, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


# -------------------------------------------#
# --------------- Data module -------------- #
# -------------------------------------------#
class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: Path,
        data_path_train: Path,
        data_path_val: Path,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        data_path_cal: Optional[Path] = None,
        cal_transform: Optional[Callable] = None,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        cal_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        cal_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        cal_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        dataset_format: str = "slice",
    ):
        super().__init__()

        # -------------------------------------------#
        # -------- sampling sanity checks ---------- #
        # -------------------------------------------#
        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Set sample_rate OR volume_sample_rate, not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError("Set val_sample_rate OR val_volume_sample_rate, not both.")
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError("Set test_sample_rate OR test_volume_sample_rate, not both.")
        if _check_both_not_none(cal_sample_rate, cal_volume_sample_rate):
            raise ValueError("Set cal_sample_rate OR cal_volume_sample_rate, not both.")

        # -------------------------------------------#
        # -------- paths and transforms ------------ #
        # -------------------------------------------#
        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.data_path_cal = data_path_cal

        self.train_transform = train_transform
        self.cal_transform = val_transform if cal_transform is None else cal_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        # -------------------------------------------#
        # -------- sampling configs ---------------- #
        # -------------------------------------------#
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path

        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.cal_sample_rate = cal_sample_rate

        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.cal_volume_sample_rate = cal_volume_sample_rate

        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.cal_filter = cal_filter

        # -------------------------------------------#
        # -------- loader configs ------------------ #
        # -------------------------------------------#
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.dataset_format = dataset_format

    # -------------------------------------------#
    # -------- dataloader builder -------------- #
    # -------------------------------------------#
    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ):

        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            raw_filter = self.train_filter

        elif data_partition == "val":
            is_train = False
            sample_rate = self.val_sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.val_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            raw_filter = self.val_filter

        elif data_partition == "cal":
            is_train = False
            sample_rate = self.cal_sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.cal_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            raw_filter = self.cal_filter

        else:
            is_train = False
            sample_rate = self.test_sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.test_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            raw_filter = self.test_filter

        # -------------------------------------------#
        # -------- dataset creation ---------------- #
        # -------------------------------------------#
        dataset_cls = VolumeDataset if self.dataset_format == "volume" else SliceDataset
        combined_dataset_cls = (
            CombinedVolumeDataset if self.dataset_format == "volume" else CombinedSliceDataset
        )

        if is_train and self.combine_train_val:
            dataset = combined_dataset_cls(
                roots=[self.data_path_train, self.data_path_val],
                transforms=[data_transform, data_transform],
                sample_rates=[sample_rate, sample_rate] if sample_rate else None,
                volume_sample_rates=[volume_sample_rate, volume_sample_rate] if volume_sample_rate else None,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_filter,
            )
        else:
            if data_partition == "test" and self.test_path is not None:
                data_path = self.test_path
            elif data_partition == "train":
                data_path = self.data_path_train
            elif data_partition == "val":
                data_path = self.data_path_val
            elif data_partition == "cal":
                if self.data_path_cal is None:
                    raise ValueError("data_path_cal must be provided to use calib_dataloader().")
                data_path = self.data_path_cal
            else:
                data_path = self.data_path_val

            dataset = dataset_cls(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_filter,
            )

        sampler = None
        if self.distributed_sampler:
            sampler = torch.utils.data.DistributedSampler(dataset) if is_train else VolumeSampler(dataset, shuffle=False)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

    # -------------------------------------------#
    # -------- prepare cache ------------------- #
    # -------------------------------------------#
    def prepare_data(self):
        if not self.use_dataset_cache_file:
            return

        test_path = self.test_path if self.test_path else self.data_path / "multicoil_test"

        paths_and_transforms = [
            (self.data_path_train, self.train_transform),
            (self.data_path_val, self.val_transform),
        ]

        if self.data_path_cal is not None:
            paths_and_transforms.append((self.data_path_cal, self.cal_transform))

        paths_and_transforms.append((test_path, self.test_transform))

        dataset_cls = VolumeDataset if self.dataset_format == "volume" else SliceDataset

        for path, transform in paths_and_transforms:
            _ = dataset_cls(
                root=path,
                transform=transform,
                sample_rate=self.sample_rate,
                volume_sample_rate=self.volume_sample_rate,
                use_dataset_cache=self.use_dataset_cache_file,
            )

    # -------------------------------------------#
    # -------- lightning hooks ----------------- #
    # -------------------------------------------#
    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, "train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, "val")

    def calib_dataloader(self):
        return self._create_data_loader(self.cal_transform, "cal")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, self.test_split)

    # -------------------------------------------#
    # -------- CLI arguments ------------------- #
    # -------------------------------------------#
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_path", type=Path, default=None)
        parser.add_argument("--data_path_train", type=Path, default=None)
        parser.add_argument("--data_path_val", type=Path, default=None)
        parser.add_argument("--data_path_cal", type=Path, default=None)
        parser.add_argument("--test_path", type=Path, default=None)
        parser.add_argument("--test_split", type=str, default="test", choices=("val", "test"))
        parser.add_argument("--sample_rate", type=float, default=None)
        parser.add_argument("--val_sample_rate", type=float, default=None)
        parser.add_argument("--cal_sample_rate", type=float, default=None)
        parser.add_argument("--test_sample_rate", type=float, default=None)
        parser.add_argument("--volume_sample_rate", type=float, default=None)
        parser.add_argument("--val_volume_sample_rate", type=float, default=None)
        parser.add_argument("--cal_volume_sample_rate", type=float, default=None)
        parser.add_argument("--test_volume_sample_rate", type=float, default=None)
        parser.add_argument("--use_dataset_cache_file", type=bool, default=True)
        parser.add_argument("--combine_train_val", type=bool, default=False)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dataset_format", type=str, default="slice", choices=("slice", "volume"))

        return parser

# -------------------------------------------#
# -------- raw sample container ------------ #
# -------------------------------------------#
class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


# -------------------------------------------#
# -------- combined slice dataset ---------- #
# -------------------------------------------#
class CombinedSliceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        roots: Sequence[Path],
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
    ):

        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError("Set slice sampling OR volume sampling, not both.")

        if transforms is None:
            transforms = [None] * len(roots)

        if sample_rates is None:
            sample_rates = [None] * len(roots)

        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)

        if not (
            len(roots)
            == len(transforms)
            == len(sample_rates)
            == len(volume_sample_rates)
        ):
            raise ValueError("roots/transforms/sample_rates mismatch")

        self.datasets = []
        self.raw_samples: List[FastMRIRawDataSample] = []

        for i in range(len(roots)):
            ds = SliceDataset(
                root=roots[i],
                transform=transforms[i],
                sample_rate=sample_rates[i],
                volume_sample_rate=volume_sample_rates[i],
                use_dataset_cache=use_dataset_cache,
                dataset_cache_file=dataset_cache_file,
                num_cols=num_cols,
                raw_sample_filter=raw_sample_filter,
            )

            self.datasets.append(ds)
            self.raw_samples += ds.raw_samples

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx < len(ds):
                return ds[idx]
            idx -= len(ds)


# -------------------------------------------#
# --------------- slice dataset ------------ #
# -------------------------------------------#
class SliceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
    ):

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError("Set slice sampling OR volume sampling, not both.")

        self.root = Path(root)
        self.transform = transform
        self.dataset_cache_file = Path(dataset_cache_file)
        self.recons_key = "reconstruction_rss"  # Always multicoil

        self.raw_sample_filter = raw_sample_filter or (lambda x: True)
        self.raw_samples: List[FastMRIRawDataSample] = []

        sample_rate = 1.0 if sample_rate is None else sample_rate
        volume_sample_rate = 1.0 if volume_sample_rate is None else volume_sample_rate

        # -------------------------------------------#
        # -------- load cache if available --------- #
        # -------------------------------------------#
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        if dataset_cache.get(self.root) is None or not use_dataset_cache:
            for fname in sorted(self.root.iterdir()):
                metadata, num_slices = self._retrieve_metadata(fname)

                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        self.raw_samples.append(raw_sample)

            if use_dataset_cache:
                dataset_cache[self.root] = self.raw_samples
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            self.raw_samples = dataset_cache[self.root]

        # -------------------------------------------#
        # -------- slice subsampling --------------- #
        # -------------------------------------------#
        if sample_rate < 1.0:
            random.shuffle(self.raw_samples)
            n = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:n]

        elif volume_sample_rate < 1.0:
            vol_names = sorted({rs.fname.stem for rs in self.raw_samples})
            random.shuffle(vol_names)
            n = round(len(vol_names) * volume_sample_rate)
            selected = set(vol_names[:n])

            self.raw_samples = [
                rs for rs in self.raw_samples if rs.fname.stem in selected
            ]

        if num_cols:
            self.raw_samples = [
                rs
                for rs in self.raw_samples
                if rs.metadata["encoding_size"][1] in num_cols
            ]

    # -------------------------------------------#
    # -------- retrieve metadata --------------- #
    # -------------------------------------------#
    def _retrieve_metadata(self, fname: Path):

        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )

            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            center = int(et_query(et_root, lims + ["center"]))
            maximum = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - center
            padding_right = padding_left + maximum

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    # -------------------------------------------#
    # -------- dataset API --------------------- #
    # -------------------------------------------#
    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx: int):

        fname, slice_ind, metadata = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][slice_ind]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][slice_ind] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            return kspace, mask, target, attrs, fname.name, slice_ind

        return self.transform(kspace, mask, target, attrs, fname.name, slice_ind)

# -------------------------------------------#
# ------------ volume sampler -------------- #
# -------------------------------------------#
class VolumeSampler(Sampler):

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        # -------------------------------------------#
        # -------- distributed world setup --------- #
        # -------------------------------------------#
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package not available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package not available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # -------------------------------------------#
        # -------- collect volume names ------------ #
        # -------------------------------------------#
        self.all_volume_names = sorted(
            set(str(sample[0]) for sample in self.dataset.raw_samples)
        )

        # -------------------------------------------#
        # -------- split volumes per rank ---------- #
        # -------------------------------------------#
        self.all_volumes_split: List[List[str]] = []
        for r in range(self.num_replicas):
            self.all_volumes_split.append(
                [
                    self.all_volume_names[i]
                    for i in range(r, len(self.all_volume_names), self.num_replicas)
                ]
            )

        # -------------------------------------------#
        # -------- slice indices per rank ---------- #
        # -------------------------------------------#
        rank_indices: List[List[int]] = [[] for _ in range(self.num_replicas)]

        for idx, raw_sample in enumerate(self.dataset.raw_samples):
            vol_name = str(raw_sample[0])
            for r in range(self.num_replicas):
                if vol_name in self.all_volumes_split[r]:
                    rank_indices[r].append(idx)
                    break

        # -------------------------------------------#
        # -------- equalize sample counts ---------- #
        # -------------------------------------------#
        self.num_samples = max(len(v) for v in rank_indices)
        self.total_size = self.num_samples * self.num_replicas
        self.indices = rank_indices[self.rank]

    # -------------------------------------------#
    # -------------- iterator ------------------ #
    # -------------------------------------------#
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in order]
        else:
            indices = list(self.indices)

        # -------------------------------------------#
        # -------- pad to equal length ------------ #
        # -------------------------------------------#
        repeat_times = self.num_samples // len(indices)
        indices = indices * repeat_times
        indices += indices[: self.num_samples - len(indices)]

        assert len(indices) == self.num_samples
        return iter(indices)

    # -------------------------------------------#
    # -------------- length -------------------- #
    # -------------------------------------------#
    def __len__(self):
        return self.num_samples

    # -------------------------------------------#
    # -------- epoch for shuffling ------------- #
    # -------------------------------------------#
    def set_epoch(self, epoch: int):
        self.epoch = epoch


# -------------------------------------------#
# -------- volume sample container --------- #
# -------------------------------------------#
class FastMRIRawVolumeSample(NamedTuple):
    fname: Path
    metadata: Dict[str, Any]


# -------------------------------------------#
# -------------- volume dataset ------------ #
# -------------------------------------------#
class VolumeDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path] = "dataset_cache.pkl",
        raw_sample_filter: Optional[Callable] = None,
    ):

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError("Set sample_rate OR volume_sample_rate, not both.")

        self.root = Path(root)
        self.transform = transform
        self.dataset_cache_file = Path(dataset_cache_file)
        self.recons_key = "reconstruction_rss"  # Always multicoil

        self.raw_sample_filter = raw_sample_filter or (lambda x: True)
        self.raw_samples: List[FastMRIRawVolumeSample] = []

        sample_rate = 1.0 if sample_rate is None else sample_rate
        volume_sample_rate = 1.0 if volume_sample_rate is None else volume_sample_rate

        # Use a separate cache key from SliceDataset.
        # SliceDataset caches one entry per slice, while VolumeDataset caches one entry per file.
        cache_key = ("volume", str(self.root.resolve()))

        # -------------------------------------------#
        # -------- load cache if available --------- #
        # -------------------------------------------#
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        if dataset_cache.get(cache_key) is None or not use_dataset_cache:
            for fname in sorted(self.root.iterdir()):
                metadata, _ = self._retrieve_metadata(fname)

                raw_sample = FastMRIRawVolumeSample(fname, metadata)
                if self.raw_sample_filter(raw_sample):
                    self.raw_samples.append(raw_sample)

            if use_dataset_cache:
                dataset_cache[cache_key] = self.raw_samples
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            self.raw_samples = dataset_cache[cache_key]

        # -------------------------------------------#
        # -------- volume subsampling ------------- #
        # -------------------------------------------#
        effective_sample_rate = min(sample_rate, volume_sample_rate)

        if effective_sample_rate < 1.0:
            random.shuffle(self.raw_samples)
            n = round(len(self.raw_samples) * effective_sample_rate)
            self.raw_samples = self.raw_samples[:n]

    # -------------------------------------------#
    # -------- retrieve metadata --------------- #
    # -------------------------------------------#
    def _retrieve_metadata(self, fname: Path):

        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )

            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            center = int(et_query(et_root, lims + ["center"]))
            maximum = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - center
            padding_right = padding_left + maximum

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                "num_slices": num_slices,
                **hf.attrs,
            }

        return metadata, num_slices

    # -------------------------------------------#
    # -------- dataset API --------------------- #
    # -------------------------------------------#
    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx: int):

        fname, metadata = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][()]  # [D, C, H, W]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][()] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            return kspace, mask, target, attrs, fname.name

        return self.transform(kspace, mask, target, attrs, fname.name)
    

# -------------------------------------------#
# -------- combined volume dataset --------- #
# -------------------------------------------#
class CombinedVolumeDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        roots: Sequence[Path],
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path] = "dataset_cache.pkl",
        raw_sample_filter: Optional[Callable] = None,
    ):

        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError("Set sample sampling OR volume sampling, not both.")

        if transforms is None:
            transforms = [None] * len(roots)

        if sample_rates is None:
            sample_rates = [None] * len(roots)

        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)

        if not (
            len(roots)
            == len(transforms)
            == len(sample_rates)
            == len(volume_sample_rates)
        ):
            raise ValueError("roots/transforms/sample_rates mismatch")

        self.datasets = []
        self.raw_samples: List[FastMRIRawVolumeSample] = []

        for i in range(len(roots)):
            ds = VolumeDataset(
                root=roots[i],
                transform=transforms[i],
                sample_rate=sample_rates[i],
                volume_sample_rate=volume_sample_rates[i],
                use_dataset_cache=use_dataset_cache,
                dataset_cache_file=dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )

            self.datasets.append(ds)
            self.raw_samples += ds.raw_samples

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx < len(ds):
                return ds[idx]
            idx -= len(ds)