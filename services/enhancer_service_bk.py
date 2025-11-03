import torch, os
from PIL import Image
import torchvision.transforms as T
from modeling.inference import LitColorization

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("checkpoints_unet", "pj2-color.ckpt")

if os.path.exists(MODEL_PATH):
    model = LitColorization.load_from_checkpoint(MODEL_PATH)
    model.to(device).eval()
else:
    model = None
    print("모델이 없습니다:", MODEL_PATH)

transform_gray = T.Compose([T.Resize((128,128)), T.ToTensor()])

def colorize_image(input_path: str, save_dir: str) -> str:
    if model is None:
        raise FileNotFoundError("모델이 로드되지 않았습니다.")
    img = Image.open(input_path).convert("L")
    orig_size = img.size
    x = transform_gray(img).unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.clamp((y_pred + 1)/2, 0, 1)
    y_pred_pil = T.ToPILImage()(y_pred.squeeze(0).cpu())
    y_pred_pil = y_pred_pil.resize(orig_size, Image.BICUBIC)
    output_path = os.path.join(save_dir, "colorized_" + os.path.basename(input_path))
    y_pred_pil.save(output_path)
    return output_path
