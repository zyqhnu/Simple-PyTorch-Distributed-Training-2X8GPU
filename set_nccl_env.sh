export NCCL_DEBUG=info
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$(ip -o -4 addr show scope global | awk '{print $2}' | grep -v 'lo' | head -n 1) # use this if you have multiple NICs

# Or you can set this environment variable manually as:
# export NCCL_SOCKET_IFNAME=your_nic_name # your NIC name, use ifconfig to find out
