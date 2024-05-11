
from torchvision import transforms
import torch
import gc
import matplotlib.pyplot as plt
import numpy as np

reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])


image2tensor_transformation = transforms.Compose([
        transforms.Resize(size = (28,28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])


def tensor2image(tensor):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(tensor)


def image2tensor(image):
    image2tensor_transformation = transforms.Compose([
            transforms.Resize(size = (224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])

    return image2tensor_transformation(image)
