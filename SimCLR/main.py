from dataset import Dataset, get_color_disotrtion
import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_color_disotrtion(s=0.5):
    # s is the strength of color distortion, cifar10 = 0.5
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
    rnd_color_jitter, rnd_gray])
    return color_distort


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
    
    # cifar10 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    # test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    

    print(train_loader.dataset, test_loader.dataset)
    # train_dataset = Dataset(batch_size, kernel_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    

    labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
         # 텐서를 NumPy 배열로 변환
        if isinstance(img, torch.Tensor):
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
        elif isinstance(img, Image.Image):
            img = np.array(img)
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


    # dataloader
    # dataiter = iter(train_loader)
    # images, _ = next(dataiter)  

    # imshow(torchvision.utils.make_grid(images))

