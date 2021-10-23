
import datetime
import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.multiprocessing as mp
from .batch_generator import LMBatchFier
import torch.distributed as dist
from torch.utils.data import Sampler
import math

world_size = 4


def set_init_group(model, args):
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:6557',
                            world_size=args.world_size, timeout=datetime.timedelta(0, 60),
                            rank=args.gpu)

    print("Complete to build process in {} process for Distributed Data Parallel training".format(args.gpu))
    model.to(args.gpu)


    if args.mixed_precision:
        from apex.parallel import DistributedDataParallel as ApexDDP
        # from apex.nn.parallel import DistributedDataParallel as DDP
        model = ApexDDP(model,delay_allreduce=True)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    return model

def cleanup():
    dist.destroy_process_group()


class DynamicDistributedSampler(Sampler):
    def __init__(self, batchfier_list: list, num_replicas=None, rank=None, shuffle=True):
        """
        batchfier : list of Tensor  (e.g. [ tensor.size(16,100) ,tensor.size(16,94) tensor.size(16,87) tensor.size(16,50)...])
        """
        super(DynamicDistributedSampler, self).__init__(batchfier_list)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.batchfier_list = batchfier_list
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.batchfier_list) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __igter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.batchfier_list), generator=g).tolist()
        else:
            indices = list(range(len(self.batchfier_list)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DynamicDistributedDataLoader:
    def __init__(self, dataset: IterableDataset, batch_size, shuffle, num_workers, collate_fn, pin_memory):
        batchfier = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                               collate_fn=collate_fn, pin_memory=pin_memory)
        self.dataset=dataset
        batch_list = [batch for batch in batchfier]
        sampler = DynamicDistributedSampler(batch_list, shuffle=False)
        self.loader = DataLoader(dataset=batch_list, sampler=sampler, batch_size=1)

    def __iter__(self):
        for data in self.loader:
            yield data
