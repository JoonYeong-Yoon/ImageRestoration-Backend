
import torch
from PIL import Image
from models.unet_colorization import LitColorization
from utils.lab_utils import preprocess_image, lab_tensor_to_rgb

def infer_image(img_path, weight_path='weights/colorization_weights.pth', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitColorization()
    state_dict = torch.load(weight_path, map_location=device)
    model.model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    L_tensor, _ = preprocess_image(img_path)
    L_tensor = L_tensor.to(device)
    with torch.no_grad():
        pred_ab = model(L_tensor)
    rgb = lab_tensor_to_rgb(L_tensor, pred_ab)
    Image.fromarray((rgb*255).astype('uint8')).save('output.png')
    print("✅ Saved output.png")

if __name__ == "__main__":
    infer_image("test_image.JPEG")
