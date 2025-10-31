
import torch, cv2, numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(img_path, target_size=320):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])
    img_t = transform(img)
    img_np = (img_t.numpy() * 255).astype(np.uint8).transpose(1,2,0)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = (lab[:,:,0:1] / 50.0) - 1.0
    Lt = torch.from_numpy(L.transpose(2,0,1).astype(np.float32)).unsqueeze(0)
    return Lt, img

def lab_tensor_to_rgb(L_tensor, ab_tensor):
    L = L_tensor.detach().cpu().numpy()
    ab = ab_tensor.detach().cpu().numpy()
    Li = (L[0,0,:,:] + 1.0) * 50.0
    abi = ab[0].transpose(1,2,0) * 128.0
    lab = np.zeros((Li.shape[0], Li.shape[1], 3), dtype=np.float32)
    lab[:,:,0] = Li
    lab[:,:,1:] = abi
    rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
