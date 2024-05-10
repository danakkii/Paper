import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms



# CIFAR10
class Dataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, kernel_size):
        # self.data = data
        # self.label = label
        # self.patch_size = patch_size
        # self.indice = indice
        self.batch_size = batch_size
        self.kernel_size = kernel_size

    def __len__(self):
              return  # location길이 만큼 
       
    def __getitem__(self, i):
        batch_size = 1024 # best
        kernel_size = 3 # cifar10 set to be 10% of the image height/width
        transforms = transforms.Compose([
              transforms.RandomCrop(256),
              transforms.color_distort(),
              transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        print(train_loader.dataset, test_loader.dataset)

        return test_loader
              

   

def get_color_disotrtion(s=0.5):
    # s is the strength of color distortion, cifar10 = 0.5
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
    rnd_color_jitter, rnd_gray])
    return color_distort


batch_size = 1024 # cifar10 best accuracy
kernel_size = 3 # cifar10 set to be 10% of the image height/width, the image 50% of the time using a Gaussian kernel
transforms = transforms.Compose([
        transforms.RandomCrop(224),
        get_color_disotrtion(),
        transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(train_loader.dataset, test_loader.dataset)

