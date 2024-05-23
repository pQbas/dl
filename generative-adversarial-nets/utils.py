import os
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

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


class Plotter:
    def __init__(self, nFigures):
        plt.ion()
        self.figures = [plt.figure() for _ in range(nFigures-1)]


    def plot(self, data2plot): # information2plot -> (data, datatype)
 
        for (index, data, dataType) in data2plot:
            plt.figure(index)
            if 'plot' == dataType: plt.plot(data)
            elif 'imshow' == dataType: plt.imshow(data)
            else: print("Doesn't exist the type of plot selected")

        plt.draw()
        plt.pause(0.001)

    def stop(self):
        print('Press key "q" to exit')
        plt.show(block=True)






