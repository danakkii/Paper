from dataset import Dataset, get_color_disotrtion
import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def get_color_disotrtion(s=0.5):
    # s is the strength of color distortion, cifar10 = 0.5
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
    rnd_color_jitter, rnd_gray])
    return color_distort

if __name__=="__main__":


    batch_size = 1024 # cifar10 best accuracy
    kernel_size = 3 # cifar10 set to be 10% of the image height/width, the image 50% of the time using a Gaussian kernel
    s = 0.5 # is the strength of color distortion.

    transforms = transforms.Compose([
            transforms.RandomCrop(32),# flip and resize to 32x32
            get_color_disotrtion(s=0.5),
            transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
            transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(train_loader.dataset, test_loader.dataset)
    # train_dataset = Dataset(batch_size, kernel_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    

    

    # 이미지를 보여주기 위한 함수
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # 학습용 이미지 배치를 가져옵니다.
    dataiter = iter(train_loader)
    images, _ = next(dataiter)  # 다음 배치를 가져옵니다.

    # 이미지를 보여줍니다.
    imshow(torchvision.utils.make_grid(images))

