from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from veomni.data.dataset import IterativeDataset, MappingDataset
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.dist_utils import main_process_first


def build_local_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
    source_name: Optional[str] = None,
):
    parallel_state = get_parallel_state()
    dataset = load_from_disk(Path(data_path) / namespace)
    dataset = dataset.shuffle(seed=seed)

    if transform:
        transform = partial(transform, source_name=source_name)
    return MappingDataset(dataset, transform=transform)


def build_hf_dataset(
    path: str,
    config_name: Optional[str] = None,
    transform: Optional[Callable] = None,
    namespace: str = "train",
    seed: int = 42,
    source_name: Optional[str] = None,
):

    parallel_state = get_parallel_state()

    dataset = load_dataset(path, config_name, split=namespace, num_proc=8)
    dataset = dataset.shuffle(seed=seed)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    if transform:
        transform = partial(transform, source_name=source_name)
    return MappingDataset(dataset, transform=transform)
