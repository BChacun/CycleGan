import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(config):
    """Builds and returns Dataloader for ANIME and HUMAN dataset."""

    transformhuman = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transformanime = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    human = datasets.ImageFolder(root=config.human_path, transform=transformhuman)
    anime = datasets.ImageFolder(root=config.anime_path, transform=transformanime)

    human_loader = torch.utils.data.DataLoader(dataset=human,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    anime_loader = torch.utils.data.DataLoader(dataset=anime,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return human_loader, anime_loader
