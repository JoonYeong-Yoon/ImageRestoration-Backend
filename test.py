import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from network.models import uformer
from network.colorization_model import ColorizationModel

def pad_to_divisible(x, div=16):
    _, _, h, w = x.size()
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    return F.pad(x, (0, pad_w, 0, pad_h)), h, w

colorization_model = ColorizationModel()
pil_data = Image.open(r"e:\Advanced_Project\TEST6.JPEG")
res = colorization_model._process_image(pil_data)
res.save(r"E:\Advanced_Project/out.png")

restoration_model = uformer.UNet(dim=32)
weight_file_path = "network/weights/damageRestoration/uformer_best.pth"
restoration_weights = torch.load(weight_file_path,map_location="cpu")
restoration_model.load_state_dict(restoration_weights)

input_path = r"e:\Advanced_Project\damaged.jpeg"
transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()
img = Image.open(input_path).convert("RGB")
orig_w, orig_h = img.size
img_tensor = transform(img).unsqueeze(0)

padded_tensor, orig_h, orig_w = pad_to_divisible(img_tensor, div=16)

# ====== 복원 ======
with torch.no_grad():
    output_tensor = restoration_model(padded_tensor)

# crop 원래 크기로
output_tensor = output_tensor[:, :, :orig_h, :orig_w]

# ====== 후처리 및 저장 ======
output_img = output_tensor.squeeze(0).cpu()
output_img = to_pil(output_img.clamp(0, 1))
output_path = r"e:\Advanced_Project\out2.png"
output_img.save(output_path)