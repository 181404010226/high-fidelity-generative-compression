import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_paths = []

        print(f"Initializing dataset from folder: {root}")

        # 遍历目录并收集所有PNG图片
        for file in os.listdir(self.root):
            if file.lower().endswith('.png'):
                self.img_paths.append(os.path.join(self.root, file))

        print(f"Number of images found: {len(self.img_paths)}")
        if len(self.img_paths) > 0:
            print(f"First image path: {self.img_paths[0]}")
        else:
            print("No images found!")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_dataloader(root, batch_size, crop_size, normalize=False, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
    ])
    if normalize:
        transform.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    dataset = ImageFolder(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader