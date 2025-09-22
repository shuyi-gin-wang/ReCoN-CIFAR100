import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from resnet18 import ResNet, BasicBlock


CIFAR100_CLASS_TO_SUPER = [4,1,14,8,0,6,7,7,18,3,3,14,9,18,7,11,3,9,7,11,6,11,5,10,7,6,13,15,3,15,0,11,1,10,12,14,16,9,11,5,5,19,8,8,15,13,14,17,18,10,16,4,17,4,2,0,17,4,18,17,10,3,2,12,12,16,12,1,9,19,2,10,0,1,16,12,9,13,15,13,16,19,2,4,6,19,5,5,8,19,18,1,2,15,6,0,17,8,14,13]

FINE_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
    'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster',
    'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
    'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
    'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
    'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
    'wolf', 'woman', 'worm'
]

SUPER_LABELS = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
    'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
]

CLASS_TO_NAME = {i: name for i, name in enumerate(FINE_LABELS)}
SUPER_TO_NAME = {i: name for i, name in enumerate(SUPER_LABELS)}

def labels(y: int):
    return (y, CIFAR100_CLASS_TO_SUPER[y])


def get_loaders(data_dir, batch_size):
    train_augmentation = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),]) # cifar 100 specific stats
    test_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    train_ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_augmentation, target_transform=labels)
    test_ds = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_norm, target_transform=labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def resnet18_cifar100(drop_rate=0.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], class_num=100, superclass_num=20, drop_rate=drop_rate)

@torch.no_grad()
def load_model_from_pt(pt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18_cifar100().to(device)
    ckpt = torch.load(pt_path, map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model, device
