# import os
# import torch
# from PIL import Image
# from torchvision import transforms
# import torch.nn.functional as F

# from network.models import uformer
# from network.colorization_model import ColorizationModel

# def pad_to_divisible(x, div=16):
#     _, _, h, w = x.size()
#     pad_h = (div - h % div) % div
#     pad_w = (div - w % div) % div
#     return F.pad(x, (0, pad_w, 0, pad_h)), h, w

# colorization_model = ColorizationModel()
# pil_data = Image.open(r"e:\Advanced_Project\TEST6.JPEG")
# res = colorization_model._process_image(pil_data)
# res.save(r"E:\Advanced_Project/out.png")

# restoration_model = uformer.UNet(dim=32)
# weight_file_path = "network/weights/damageRestoration/uformer_best.pth"
# restoration_weights = torch.load(weight_file_path,map_location="cpu")
# restoration_model.load_state_dict(restoration_weights)

# input_path = r"e:\Advanced_Project\damaged.jpeg"
# transform = transforms.ToTensor()
# to_pil = transforms.ToPILImage()
# img = Image.open(input_path).convert("RGB")
# orig_w, orig_h = img.size
# img_tensor = transform(img).unsqueeze(0)

# padded_tensor, orig_h, orig_w = pad_to_divisible(img_tensor, div=16)

# # ====== 복원 ======
# with torch.no_grad():
#     output_tensor = restoration_model(padded_tensor)

# # crop 원래 크기로
# output_tensor = output_tensor[:, :, :orig_h, :orig_w]

# # ====== 후처리 및 저장 ======
# output_img = output_tensor.squeeze(0).cpu()
# output_img = to_pil(output_img.clamp(0, 1))
# output_path = r"e:\Advanced_Project\out2.png"
# output_img.save(output_path)

# # timm, einops
# from network.models import uformer_source

# """
# python3 ./train/train_denoise.py --arch Uformer_B --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir ../datasets/denoising/sidd/train --env _0706 \
#     --val_dir ../datasets/denoising/sidd/val --save_dir ./logs/ \
#     --dataset sidd --warmup 
# model_restoration = Uformer(
#     img_size=opt.train_ps,
#     embed_dim=32,
#     win_size=8,
#     token_projection='linear',
#     token_mlp='leff',
#     depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
#     modulator=True,dd_in=opt.dd_in)  
# """

# model_restoration = uformer_source.Uformer(
#     img_size=128,embed_dim=32,
#     win_size=8,
#     token_projection='linear',
#     token_mlp='leff',
#     depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
#     modulator=True,
#     dd_in=3)  
# weight_file_path = "network/weights/damageRestoration/uformer_B.pth"
# import torch
# restoration_weights = torch.load(weight_file_path,map_location="cpu")
# model_restoration
# restoration_weights["state_dict"].keys()
# model_restoration.load_state_dict(restoration_weights)

# pretrained_dict = restoration_weights["state_dict"]
# model_restoration.load_state_dict(pretrained_dict)

# model_dict = model_restoration.state_dict()
# matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

# from collections import OrderedDict
# state_dict = restoration_weights["state_dict"]
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] if 'module.' in k else k
#     new_state_dict[name] = v
# new_state_dict.keys()
# restoration_model.load_state_dict(new_state_dict)
# for name, param in restoration_model.named_parameters():
#     print(name, param.shape)


# restoration_model.load_state_dict(new_state_dict)


from network.restore_model_unet import RestoreUNetModel
from PIL import Image
import os, torch
weight_path = os.path.join("network", "weights", "damageRestoration", "last_epoch_model_epoch_3.pth")

unet = RestoreUNetModel()
img = Image.open("./exports/processed/damaged2.jpeg").convert("RGB")

pred = unet.restore_with_unet(img)
pred.save("./exports/processed/damaged2-check.jpeg")

unet.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=3,
                activation='tanh'
            ).to("cpu")
checkpoint = torch.load(weight_path, map_location="cpu")
state_dict = checkpoint["generator_state_dict"]
unet.model.load_state_dict(state_dict, strict=False)

import segmentation_models_pytorch as smp
from torchvision import transforms
unet = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=3,
                activation='tanh'
            ).to("cpu")
checkpoint = torch.load(weight_path, map_location="cpu")
state_dict = checkpoint["generator_state_dict"]
unet.load_state_dict(state_dict, strict=False)


img = Image.open("./exports/processed/damaged2.jpeg").convert("RGB")
original_width, original_height = img.size
inference_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
damaged_image_tensor = inference_transform(img).unsqueeze(0).to("cpu") # Add batch dimension and move to device
with torch.no_grad():
    restored_image_tensor = unet(damaged_image_tensor)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
restored_image_tensor = inv_normalize(restored_image_tensor.squeeze(0).cpu()) # Remove batch dim and move to CPU
restored_image_tensor = torch.clamp(restored_image_tensor, 0, 1) # Clamp values to [0, 1]

# Convert tensor to PIL Image and resize to original size
restored_image_pil = transforms.ToPILImage()(restored_image_tensor)
restored_image_pil = restored_image_pil.resize((original_width, original_height)) # Resize back to original size

