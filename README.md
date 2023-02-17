!--
 * @Author: Steven Basart
 * @Date: 2023-02-16 18:31:00
 * @Description: In User Settings Edit
 * @FilePath: \PyTorch-Distributed-Training\README.md
 -->
# PyTorch-Distributed-Training
Example of PyTorch DistributedDataParallel

## Installation

Code was tested with the following with CUDA version 11.6

torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1

## Single machine multi gpu
```
python -m torch.distributed.launch --nproc_per_node=ngpus --master_port=29500 simple_dist_mnist.py ...
```

## Multi machine multi gpu
Suppose we have two machines and each machine has 8 gpus.

In multi machine multi gpu situation, you have to choose a machine to be main node.

We named the machines A and B, and set A to be main node.

To get the ip of the host A the main node run `hostname -i`.

command to run at A

```
python -m torch.distributed.launch --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=A_ip_address --master_port=29500 simple_dist_mnist.py --world_size=16
```

command to run at B

```
python -m torch.distributed.launch --nproc_per_node=8 --nnode=2 --node_rank=1 --master_addr=A_ip_address --master_port=29500 simple_dist_mnist.py --world_size=16
```

Note that this version expects `--local_rank` to be an arg to the function.
