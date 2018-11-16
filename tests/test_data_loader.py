import unittest
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from src.data_loader import split_omniglot_characters
from src.data_loader import load_imagenet_images
from src.data_loader import OmniglotTask
from src.data_loader import ImageNetTask
from src.data_loader import FewShotDataset
from src.data_loader import train_transformer_Omniglot
from src.data_loader import eval_transformer_Omniglot
from src.data_loader import train_transformer_ImageNet
from src.data_loader import eval_transformer_ImageNet
from src.data_loader import fetch_dataloaders


class test_data_loader(unittest.TestCase):
    def setUp(self):
        # Define 5-way 5-shot few shot task.
        # TODO for miniImageNet, tieredImageNet
        self.num_classes = 5
        self.num_samples = 5
        self.num_query = 3
        data_dir = 'data/Omniglot/'
        data_dir_imagenet = 'data/miniImageNet/'
        SEED = 0

        train_chars, test_chars = split_omniglot_characters(data_dir, SEED)
        self.train_chars = train_chars
        self.test_chars = test_chars
        self.task = OmniglotTask(self.train_chars, self.num_classes,
                                 self.num_samples, self.num_query)

        train_images, test_images = load_imagenet_images(data_dir_imagenet)
        self.train_images = train_images
        self.test_images = test_images
        self.task_mini_image = ImageNetTask(self.train_images,
                                            self.num_classes, self.num_samples,
                                            self.num_query)

    def test_split_omniglot_characters(self):
        self.assertTrue(len(self.train_chars) == 1200)
        self.assertTrue(len(self.test_chars) == 423)

    def test_task_generator_Omniglot(self):
        train_roots = self.task.train_roots
        test_roots = self.task.test_roots
        train_labels = self.task.train_labels
        test_labels = self.task.test_labels

        self.assertTrue(
            len(train_roots) == self.num_classes * self.num_samples)
        self.assertTrue(len(test_roots) == self.num_classes * self.num_query)

        for root, label in zip(train_roots, train_labels):
            print("filename {} and label {}".format(root, label))
        for root, label in zip(test_roots, test_labels):
            print("filename {} and label {}".format(root, label))

    def test_task_generator_miniImageNet(self):
        train_roots = self.task_mini_image.train_roots
        test_roots = self.task_mini_image.test_roots
        train_labels = self.task_mini_image.train_labels
        test_labels = self.task_mini_image.test_labels

        self.assertTrue(
            len(train_roots) == self.num_classes * self.num_samples)
        self.assertTrue(len(test_roots) == self.num_classes * self.num_query)

        for root, label in zip(train_roots, train_labels):
            print("filename {} and label {}".format(root, label))
        for root, label in zip(test_roots, test_labels):
            print("filename {} and label {}".format(root, label))

    @unittest.skip("Not implemented yet")
    def test_task_generator_tieredImageNet(self):
        pass

    def test_dataset_Omniglot(self):
        train_roots = self.task.train_roots
        test_roots = self.task.test_roots
        train_labels = self.task.train_labels
        test_labels = self.task.test_labels

        dataset_train = FewShotDataset(train_roots, train_labels,
                                       train_transformer_Omniglot)
        dataset_test = FewShotDataset(test_roots, test_labels,
                                      eval_transformer_Omniglot)

        image_tr, label_tr = dataset_train[0]
        image_te, label_te = dataset_test[0]
        self.assertTrue(image_tr.size() == (1, 28, 28))
        self.assertTrue(image_te.size() == (1, 28, 28))

        image_tr = transforms.ToPILImage()(image_tr)
        image_te = transforms.ToPILImage()(image_te)
        # image_tr.show()
        # image_te.show()

    def test_dataset_miniImageNet(self):
        train_roots = self.task_mini_image.train_roots
        test_roots = self.task_mini_image.test_roots
        train_labels = self.task_mini_image.train_labels
        test_labels = self.task_mini_image.test_labels

        dataset_train = FewShotDataset(train_roots, train_labels,
                                       train_transformer_ImageNet)
        dataset_test = FewShotDataset(test_roots, test_labels,
                                      eval_transformer_ImageNet)

        image_tr, label_tr = dataset_train[0]
        image_te, label_te = dataset_test[0]
        self.assertTrue(image_tr.size() == (3, 84, 84))
        self.assertTrue(image_te.size() == (3, 84, 84))

        image_tr = transforms.ToPILImage()(image_tr)
        image_te = transforms.ToPILImage()(image_te)
        image_tr.show()
        image_te.show()

    @unittest.skip("Not implemented yet")
    def test_dataset_tieredImageNet(self):
        pass

    def test_fetch_dataloaders(self):
        dataloaders = fetch_dataloaders(['train', 'test'], self.task)
        dl_train = dataloaders['train']
        dl_test = dataloaders['test']

        N = self.num_classes * self.num_samples
        T = self.num_classes * self.num_query

        for i, (sup_samples, sup_labels) in enumerate(dl_train):
            self.assertTrue(sup_samples.size() == (N, 1, 28, 28))
        self.assertTrue(i == 0)

        for i, (que_samples, que_labels) in enumerate(dl_test):
            self.assertTrue(que_samples.size() == (T, 1, 28, 28))
        self.assertTrue(i == 0)


if __name__ == '__main__':
    unittest.main()