import os
import torch
import multiprocessing

import torch.distributed as t_dist


def dist_init(port=2333):
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    rank = int(os.environ['SLURM_PROCID'])
    world_size = os.environ['SLURM_NTASKS']
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = world_size
    os.environ['RANK'] = str(rank)
    
    t_dist.init_process_group(backend='nccl')
    
    return rank, int(world_size), gpu_id
