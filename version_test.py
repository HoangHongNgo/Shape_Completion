import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 打印 GPU 设备数量
    print(torch.cuda.device_count())
    # 打印当前 GPU 设备的名称
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU available.")


