import torch
from models.backbones import LiteHRNet, LiteHRNet_pytorch
from configs.top_down.lite_hrnet.coco import custom_18_384x288 as config

config_path = "configs/top_down/lite_hrnet/coco/litehrnet_18_coco_384x288.py"
device = torch.device('cuda')
architecture = LiteHRNet_pytorch


config.model["pretrained"] = None
extra = config.model["backbone"]["extra"]
norm_eval = False
with_cp = False

model = architecture(extra=extra, in_channels=3, norm_eval=norm_eval,
                     with_cp=with_cp)

model.to(device)
model.eval()
print("init ok")

scale = 25 # my laptop 25
x = torch.rand(1, 3, 32*scale, 32*2*scale).to(device)
y = model(x)[0]
print(f"output shape: {y.size()}")
print("inference ok")


