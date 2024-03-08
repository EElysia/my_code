from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
import torch
from random import randint


class My_Dataset(data.Dataset):
    def __init__(self, folder, image_name, mask_ratio=0.25, iteration=1000):
        super().__init__()
        self.folder = folder
        self.path = os.path.join(folder, image_name)
        self.mask_ratio = mask_ratio
        self.iteration = iteration
        self.img = Image.open(self.path).convert('RGB')
        self.scale = min(250 / max([self.img.size[0], self.img.size[1]]), 1)
        self.h = int(self.img.size[0] * self.scale)
        self.h = self.h-self.h % 8
        self.w = int(self.img.size[1] * self.scale)
        self.w = self.w - self.w % 8

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((self.w, self.h), scale=(0.85, 1.0)),
            # transforms.Resize((self.w, self.h)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        # dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels", "masks"])

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):
        img = self.transform(self.img)
        masked_image, pos = random_mask_image(img, mask_ratio=self.mask_ratio)
        return img, masked_image, pos

def random_mask_image(input_image, mask_ratio=0.5):
    channel, h, w = input_image.shape
    new_h, new_w = int(h * mask_ratio), int(w * mask_ratio)
    mask = torch.zeros_like(input_image)


    start_h = randint(0, h-new_h+1)
    start_w = randint(0, w-new_w+1)
    mask[:, start_h:start_h+new_h, start_w:start_w+new_w] = 1
    masked_image = input_image * mask
    pos = torch.Tensor([start_w/w, new_w/w, start_h/h, new_h/h])
    return masked_image, pos

def transback(data):
    return data / 2 + 0.5
