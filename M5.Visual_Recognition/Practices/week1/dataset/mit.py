import os

from PIL import Image
from torch.utils.data import Dataset


# from torchvision.transforms import transforms

# # define the data directory and transform
# data_dir = 'path/to/mit_indoor_scenes'
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # create the training and testing datasets
# train_dataset = MITDataset(data_dir, split='train', transform=transform)
# test_dataset = MITDataset(data_dir, split='test', transform=transform)

# # create the training and testing data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)


class MITDataset(Dataset):
    def __init__(self, data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train', transform=None):
        self.data_dir = os.path.join(data_dir, split_name)
        self.transform = transform
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []
        for target in self.classes:
            d = os.path.join(self.data_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
