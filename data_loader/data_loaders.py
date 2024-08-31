
import torch
from torchvision import datasets, transforms

class BCDataLoader:
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers = num_workers

        # Data transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert PIL Image to Tensor before other transformations
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._setup()

    def _setup(self):
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.valid_data_loader = None  # Implement validation split if needed

    def split_validation(self):
        if self.validation_split > 0.0:
            # Implement splitting logic here
            pass
        return self.valid_data_loader

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return iter(self.data_loader)
