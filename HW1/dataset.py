import torch
import idx2numpy
from torch.utils.data import Dataset


class FashionMNISTDataSet(Dataset):
    def __init__(self, images_file_path, labels_file_path):
        self.images = idx2numpy.convert_from_file(images_file_path)
        self.labels = idx2numpy.convert_from_file(labels_file_path)
        self.num_of_samples = len(self.images)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        img_tensor = torch.from_numpy(self.images[idx])
        label = self.labels[idx]

        return {'label': label, 'image': img_tensor}
