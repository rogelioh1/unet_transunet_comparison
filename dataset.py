#dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class isic_data_loader(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
    
        self.image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]

        # Modified to only include regular image files and exclude superpixels
        self.image_files = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.endswith('.jpg') and not f.endswith('_superpixels.jpg')
        ]
        self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]


    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path and ID
        img_path = self.image_files[idx]
        img_id = self.image_ids[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Find corresponding mask (png format for masks) 
        mask_path = os.path.join(self.mask_dir, f"{img_id}_segmentation.png")
        
        # Load mask
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default transform for mask if none provided
            # Convert to tensor and binarize (0 or 1)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Binarize the mask
        
        return image, mask
    

def get_data_loaders(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, batch_size=16, img_size=256):
    # Define transformations for images
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transforms for masks
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = isic_data_loader(train_img_dir, train_mask_dir, train_transform, mask_transform=mask_transform)
    test_dataset = isic_data_loader(test_img_dir, test_mask_dir, test_transform, mask_transform=mask_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader