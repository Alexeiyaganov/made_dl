import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MyDataset(Dataset):

    def __init__(self, data_dir, image_fns):
        self.data_dir = data_dir
        self.image_fns = image_fns

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.data_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)

        text = image_fn.split(".")[0]
        return image, text

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)
