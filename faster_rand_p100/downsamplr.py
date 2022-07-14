import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def load_patch(path):
    lena = Image.open(path)
    #lena = lena.resize((patch_size[0], patch_size[1]))
    lena = transforms.ToTensor()(lena)
    lena = lena.unsqueeze(0)
    return lena

def save_tensor_img(path, tensor):
    pil_image = transforms.ToPILImage()(tensor)
    pil_image.save(path)

patch_p = 'faster_rand_p100_123800_down.png'
patch = load_patch(patch_p)
transformed_patch = F.adaptive_avg_pool2d(patch,
            (50, 50))
save_tensor_img('faster_rand_p50_123800_down.png',transformed_patch[0])