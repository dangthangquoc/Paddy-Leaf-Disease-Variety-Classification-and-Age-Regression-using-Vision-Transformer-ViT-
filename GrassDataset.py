from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import cv2

class GrassDataset(Dataset):
    def __init__(self, root, is_train):
        if is_train:
            data_paths = os.path.join(root, 'train_images')
        else:
            data_paths = os.path.join(root, 'test_images')
        self.categories = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight','blast','brown_spot','dead_heart','downy_mildew', 'hispa','normal','tungro']
        self.image_paths = []
        self.labels = []
        for category in self.categories:
            subdir_path = os.path.join(data_paths, category)
            for filename in os.listdir(subdir_path):
                self.image_paths.append(os.path.join(subdir_path, filename))
                self.labels.append(category)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    train_dataset = GrassDataset(root=r"D:\COSC2753_A2_MachineLearning", is_train=True)
    image, label = train_dataset[1024]
    cv2.imshow(label, image)
    cv2.waitKey(0)