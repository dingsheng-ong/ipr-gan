import torch

from PIL import Image
from torchvision import transforms


def create_watermark_mask(watermark_img, wtmk_size=16, norm=False, device=torch.device('cpu')):
    if type(wtmk_size) in [int, float]: wtmk_size = [wtmk_size, ] * 2
    
    watermark_img = Image.open(watermark_img).convert("RGBA")
    watermark_img = transforms.Resize(wtmk_size)(watermark_img)

    watermark = Image.new("RGBA", wtmk_size, (0, ) * 4)
    watermark.paste(watermark_img, (0, 0), mask=watermark_img)

    watermark = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, ] * 3 + [0], [0.5, ] * 3 + [1])
    ])(watermark)

    mask = (watermark[3].unsqueeze(0) == 0).repeat(3, 1, 1).float()
    watermark = watermark[:3] * watermark[3]

    if not norm: watermark = watermark * 0.5 + 0.5

    return {'object': watermark.to(device), 'mask': mask.to(device)}
