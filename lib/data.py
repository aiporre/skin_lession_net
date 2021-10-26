from torch.utils.data import Dataset
import os
import glob
from matplotlib import image
import pandas as pd

num_labels = 11

mean = (0.6432, 0.5252, 0.4514)
std = (0.1559, 0.1856, 0.1889)

labels_to_num = {'contact allergy': 0, \ 
        'bland': 1, \
        'pyoderma': 2, \
        'vasculitis': 3, \
        'Contact allergy': 4, \
        'Infection': 5, \
        'Necrosis': 6, \
        'necrosis': 7, \
        'malignant': 8, \
        'Bland': 9, \}

# pre-calculate the unique labels in the dataset:
def read_annotations(root):
    annotation_file = os.path.join(root, "annotations.csv")
    assert os.path.exists(annotation_file), "annotation file doesn't exists."
    annotation_data = pd.read_csv(annotation_file)
    return annotation_data

def parse_labels(annotation_data):
    labels = annotation_data["Percentages of diagnoses "]
    def parsing_fcn(label):
        tokens = label.split(",")
        tokens = [t.strip() for t in tokens]
        tokens = [t.split("%") for t in tokens]
        tokens = [[int(t[0]),t[1].strip()] for t in tokens]
        return tokens
    labels = list(map(parsing_fcn, labels))
    return labels

class SkinImages(Dataset):
    def __init__(self, root, transform = None):
        images_dir = os.path.join(root,"images", "*.JPG")
        image_files = glob.glob(os.path.join(images_dir))
        data = read_annotations(root)
        label_img_names = list(data['Image Capture'])
        labels = parse_labels(data)
        # filter thei images names in the annotations to the actual images in the path
        for image_name in label_img_names.copy():
            image_name_as_filename = os.path.join(os.path.split(images_dir)[0], image_name + ".JPG")
            if image_name_as_filename not in image_files:
                indices_delete = [i for i, name in enumerate(label_img_names) if name == image_name]
                for i_delete in indices_delete:
                    label_img_names.pop(i_delete)
                    labels.pop(i_delete)
        for image_file in image_files.copy():
            image_file_name_noext, _ = os.path.splitext(os.path.basename(image_file))
            if image_file_name_noext not in label_img_names:
                i_delete = image_files.index(image_file)
                image_files.pop(i_delete)
        indices = sorted(range(len(labels)), key=lambda k: label_img_names[k])
        image_files.sort()
        self.image_files = [image_files[k] for k in indices]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = image.imread(self.image_files[index]) #Image.open(self.image_files[index])
        labels = self.labels[index]
        # takes the highest value as label.
        labels = sorted(labels, key=lambda x:x[0])
        label = labels[-1]
        label_num = labels_to_num[label[1]]
        label_num_T = torch.tensor(label_num)
        label_num = F.one_hot(label_num_T, num_classes=num_classes)
        if self.transform:
            img = self.transform(image = img)['image']
        return img, label_num



