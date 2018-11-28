# Code is based on https://github.com/floodsung/LearningToCompare_FSL
# for Omniglot
# TODO for miniImageNet
# TODO tieredImageNet

import random
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Define a training image loader for Omniglot that specifies transforms on images.
train_transformer_Omniglot = transforms.Compose([
    transforms.Resize((28, 28)),  # Pre-resizing helps SPEED-UP
    # transforms.RandomRotation(90),  # NOTE DO or DO NOT? need to be consistent
    transforms.ToTensor()
])

# Define a evaluation loader, no random rotation.
eval_transformer_Omniglot = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.ToTensor()])

# Define a training image loader for ImageNet (miniImageNet, tieredImageNet)
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize((84, 84)),
    # transforms.RandomRotation([0, 90, 180, 270]),
    transforms.ToTensor()
])

# Define a evaluation loader, no random rotation.
eval_transformer_ImageNet = transforms.Compose(
    [transforms.Resize((84, 84)),
     transforms.ToTensor()])


def split_omniglot_characters(data_dir, SEED):
    if not os.path.exists(data_dir):
        raise Exception("Omniglot data folder does not exist.")

    character_folders = [os.path.join(data_dir, family, character) \
                        for family in os.listdir(data_dir) \
                        if os.path.isdir(os.path.join(data_dir, family)) \
                        for character in os.listdir(os.path.join(data_dir, family))]
    random.seed(SEED)
    random.shuffle(character_folders)

    test_ratio = 0.2  # against total data
    val_ratio = 0.2  # against train data
    num_total = len(character_folders)
    num_test = int(num_total * test_ratio)
    num_val = int((num_total - num_test) * val_ratio)
    num_train = num_total - num_test - num_val

    train_chars = character_folders[:num_train]
    val_chars = character_folders[num_train:num_train + num_val]
    test_chars = character_folders[-num_test:]
    return train_chars, val_chars, test_chars


def load_imagenet_images(data_dir):
    """
    The datasets for miniImageNet and tieredImageNet are already splited into
    train/val/test. The method returns the lists of paths of classes as the 
    method for omniglot does.
    """
    if not os.path.exists(data_dir):
        raise Exception("ImageNet data folder does not exist.")

    train_classes = [os.path.join(data_dir, 'train', family)\
                    for family in os.listdir(os.path.join(data_dir, 'train')) \
                    if os.path.isdir((os.path.join(data_dir, 'train', family)))]
    val_classes = [os.path.join(data_dir, 'val', family)\
                     for family in os.listdir(os.path.join(data_dir, 'val')) \
                     if os.path.isdir((os.path.join(data_dir, 'val', family)))]
    test_classes = [os.path.join(data_dir, 'test', family)\
                   for family in os.listdir(os.path.join(data_dir, 'test')) \
                   if os.path.isdir((os.path.join(data_dir, 'test', family)))]

    return train_classes, val_classes, test_classes


class Task(object):
    """
    An abstract class for defining a single few-shot task.
    """

    def __init__(self, character_folders, num_classes, support_num, query_num):
        """
        Args:
            character_folders: a list of omniglot characters that the task has
            num_classes: a number of classes in a task (N-way)
            support_num: a number of support samples per each class (K-shot)
            query_num: a number of query samples per each class NOTE how to configure ??
        """
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.support_num = support_num
        self.query_num = query_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = list(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            self.train_roots += samples[c][:support_num]
            self.test_roots += samples[c][support_num:support_num + query_num]

        self.train_labels = [
            labels[self.get_class(x)] for x in self.train_roots
        ]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        # raise NotImplementedError("This is abstract class")
        return os.path.join(*sample.split('/')[:-1])


class OmniglotTask(Task):
    """
    Class for defining a single few-shot task given Omniglot dataset.
    """

    def __init__(self, *args, **kwargs):
        super(OmniglotTask, self).__init__(*args, **kwargs)


class ImageNetTask(Task):
    """
    Class for defining a single few-shot task given ImageNet dataset.
    """

    def __init__(self, *args, **kwargs):
        super(ImageNetTask, self).__init__(*args, **kwargs)


class FewShotDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, filenames, labels, transform):
        """
        Store the filenames of the images to use.
        Specifies transforms to apply on images.

        Args:
            filenames: (list) a list of filenames of images in a single task
            labels: (list) a list of labels of images corresponding to filenames
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloaders(types, task):
    """
    Fetches the DataLoader object for each type in types from task.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' 
               depending on which data is required
        task: (OmniglotTask or ImageNet) a single task for few-shot learning
        TODO params: (Params) hyperparameters
    Returns:
        dataloaders: (dict) contains the DataLoader object for each type in types
    """
    if isinstance(task, OmniglotTask):
        train_transformer = train_transformer_Omniglot
        eval_transformer = eval_transformer_Omniglot
    else:
        train_transformer = train_transformer_ImageNet
        eval_transformer = eval_transformer_ImageNet

    dataloaders = {}
    for split in ['train', 'test']:
        if split in types:
            # use the train_transformer if training data,
            # else use eval_transformer without random flip
            if split == 'train':
                train_filenames = task.train_roots
                train_labels = task.train_labels
                dl = DataLoader(
                    FewShotDataset(train_filenames, train_labels,
                                   train_transformer),
                    batch_size=len(train_filenames),  # full-batch in episode
                    shuffle=True)  # TODO args: num_workers, pin_memory
            elif split == 'test':
                test_filenames = task.test_roots
                test_labels = task.test_labels
                dl = DataLoader(
                    FewShotDataset(test_filenames, test_labels,
                                   eval_transformer),
                    batch_size=len(test_filenames),  # full-batch in episode
                    shuffle=False)
            else:
                raise NotImplementedError()
            dataloaders[split] = dl

    return dataloaders