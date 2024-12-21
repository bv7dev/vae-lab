import torch
import torchvision.transforms as T
import torchvision

def get_train_loader(input_size, dataset_name, dataset_dir, subset_size, download=False):
    transform = T.Compose([T.Resize((input_size[2], input_size[3]), T.InterpolationMode.BICUBIC), T.ToTensor()])
    if dataset_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(dataset_dir, train=True, transform=transform, download=download)
    elif dataset_name == "CelebA":
        train_dataset = torchvision.datasets.CelebA(dataset_dir, split="train", transform=transform, download=download)
    else:
        raise NameError(f"Dataset: {dataset_name} not supported!")
    
    if subset_size > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randint(0, len(train_dataset), (subset_size,)))

    return torch.utils.data.DataLoader(train_dataset, input_size[0], shuffle=True)