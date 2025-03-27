<!--
 * @Author: Yuqiang Zhang
 * @Date: 2025-03-27 11:23:00
 * @Description: In User Settings Edit
 * @FilePath: \Simple-PyTorch-Distributed-Training-2X8GPU\README.md
 -->
# PyTorch-Distributed-Training
Example of PyTorch DistributedDataParallel

## Installation

- [x] Code was tested with the CUDA 11.3, torch== 1.12.0+cu113, torchaudio==0.12.0+cu113, torchvision==0.13.0+cu113.

    ```bash
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

- [x] Code was tested with the following with CUDA version 11.6, torch==1.13.1, torchaudio==0.13.1, torchvision==0.14.1 by 
[@xksteven](https://github.com/xksteven/Simple-PyTorch-Distributed-Training)


## set_env

```bash
export NCCL_DEBUG=info
export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=your_nic_name # your NIC name, use ifconfig to find out
export NCCL_SOCKET_IFNAME=$(ip -o -4 addr show scope global | awk '{print $2}' | grep -v 'lo' | head -n 1)
```

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
python -m torch.distributed.launch --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=A_ip_address --master_port=29500 simple_dist_mnist.py
```

command to run at B

```
python -m torch.distributed.launch --nproc_per_node=8 --nnode=2 --node_rank=1 --master_addr=A_ip_address --master_port=29500 simple_dist_mnist.py
```

Note that this version __DONOT NEED__ `--local_rank` to be an arg to the function.

# Issues

## 1. NCCL Error
```
Traceback (most recent call last):
  File "simple_dist_mnist.py", line 106, in <module>
    main()
  File "/home/xxx/miniconda3/envs/test_ddp/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 345, in wrapper
    return f(*args, **kwargs)
  File "simple_dist_mnist.py", line 40, in main
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
  File "/home/xxx/miniconda3/envs/test_ddp/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 646, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/home/xxx/miniconda3/envs/test_ddp/lib/python3.8/site-packages/torch/distributed/utils.py", line 89, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1191, unhandled system error, NCCL version 2.10.3
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. It can be also caused by unexpected exit of a remote peer, you can check NCCL warnings for failure reason and see if there is connection closure by a peer.
```

If the error occurs as above, run the set_nccl_env.sh on each machine.



## 2. Download MNIST dataset Error

### 1) error1: requests.exceptions.SSLError
```bash
requests.exceptions.SSLError: HTTPSConnectionPool(host='download.openmmlab.com', port=443): Max retries exceeded with url: /mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1133)')))
```
change the verify to False:
#### win
```
D:\anaconda\envs\openmmlab\Lib\site-packages\requests\sessions.py
D:\anaconda\envs\openmmlab\Lib\site-packages\pip\_vendor\requests\sessions.py
```
#### linux
```
/root/anaconda3/envs/openmmlab/lib/python3.9/site-packages/requests/sessions.py
```
#### chage the line 777 of session.py
```python
        #verify = merge_setting(verify, self.verify)
        verify = False
```

### 2) error2: certificate verify failed: self-signed certificate in certificate chain
when use `urllib.request.urlopen`,
find the file `/xx/xx/lib/python3.9/urllib/request.py`

change the `urlopen` function:

```python
def urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
            *, cafile=None, capath=None, cadefault=False, context=None):

    # add at the beginning
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    if not context:
        context=ctx
```

reference is [here](https://stackoverflow.com/questions/36600583/python-3-urllib-ignore-ssl-certificate-verification)