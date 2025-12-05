import torch
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("cuda.is_available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))