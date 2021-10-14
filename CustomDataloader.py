# %%
# Import the modules here!
from __future__ import print_function, division
import os
# load PIL image open function as imread
from PIL.Image import open as imread
import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from labels import Labels

# Ignore warnings
import warnings

from torchvision.transforms.transforms import Normalize, ToTensor
warnings.filterwarnings("ignore")
plt.ion() # interactive mode


"""
Inherit the proper class from PyTorch module.
"""
class CustomDataset(Dataset):
    def __init__(self, root="dataset", transform=None):
        """Define custom dataset with default location of root dataset folder"""
        # Load the paths to the images and labels into list/arrays in constructor here
        self.root = root
        self.images, self.labels = self.populate_dataset(root)
        #print("images", self.images)
        #print("labels", self.labels)
        self.transform = transform
    
    #Do NOT forget to override the correct methods!
    def __len__(self) -> int:
        """Return size of the dataset"""
        return len(self.images)

    def __getitem__(self, index):
        """To support the indexing such as dataset[i] in order to get i-th sample"""
        if torch.is_tensor(index):
            index = index.tolist()
        # load images
        img_name = self.images[index]
        image = imread(img_name)
        # load label
        lbl_name = self.labels[index]
        label = imread(lbl_name)
        # treat each sample and corresponding label as dictioary
        sample = {"image":image, "label":label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    """
    The colours mapping are defined in `labels.py` source code!
    You can import it and use the Label class in this method!
    """
    def shows(self, image, label):
        # Complete the implementation.
        # transform label from tensor into PIL image, to recover integer value of label
        # @NOTE: transofrm all tensors (image and label) to PIL Image
        label = transforms.ToPILImage()(label.squeeze_(0))
        # @NOTE: Denormalize image
        image = Normalize(
            mean=(-0.485/0.229, -0.456/0.224, -0.406/0.255), 
            std=(1/0.229, 1/0.224, 1/0.255)
        )(image.squeeze_(0))
        image = transforms.ToPILImage()(image.squeeze_(0))
        # convert PIL image to numpy array, squeeze additional dimension
        image = np.asarray(image).squeeze()
        # @NOTE: uncomment for loading tensor image
        #image = np.asarray(image.permute(0, 2, 3, 1)).squeeze()
        label = np.asarray(label).squeeze()
        # set subplot size to fixed size
        _, axs = plt.subplots(2, figsize=(15, 15))
        # show image and colorized label in one plot 
        axs[0].imshow(image)
        axs[0].set_title("Image Data")
        axs[1].imshow(Labels.colorize(label))
        axs[1].set_title("Colorized Label")

        # save image and label

    @classmethod
    def populate_dataset(cls, root) -> list:
        image_list = []
        label_list = []
        for dirname, _, filenames in os.walk(os.path.join(root, "images")):
            for filename in filenames:
                image_list.append(os.path.join(dirname, filename))
        
        for dirname, _, filenames in os.walk(os.path.join(root, "labels")):
            for filename in filenames:
                label_list.append(os.path.join(dirname, filename))
            
        return image_list, label_list


class CustomRandomCrop(object):
    """Random Crop transformation: using PyTorch implementation"""
    def __init__(self, output_size=(512, 512)) -> None:
        assert isinstance(output_size, (int, tuple))
        # if output_size is an integer, 
        # target output set to (W=output_size, H=output_size)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample) -> dict:
        """Return transformed image and corresponding transformed label"""
        image, label = sample["image"], sample["label"]
        # Get parameter of RandomCrop for applying 
        # same transformation parameter on the image and the label
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.output_size
        )
        # using functional API crop with paramter from RandomCrop
        img_trans = transforms.functional.crop(image, i, j, h, w)
        lbl_trans = transforms.functional.crop(label, i, j, h, w)

        return {"image":img_trans, "label":lbl_trans}


class CustomRandomHorizontalFlip(object):
    """Random Horizontal Flip transformation: using PyTorch implementation"""
    def __init__(self, p=1.0) -> None:
        assert isinstance(p, float)
        assert p > 0 and p <= 1
        self.p = p

    def __call__(self, sample) -> dict:
        """Return transformed image and corresponding transformed label"""
        image, label = sample["image"], sample["label"]
        trans = transforms.RandomHorizontalFlip(p=1)
        img_trans = trans(image)
        lbl_trans = trans(label)

        return {"image":img_trans, "label":lbl_trans}


class CustomNormalize(object):
    """Normalization transformation: only transform image, leave label as it was.
    Normalization parameter based on ImageNet dataset"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
        assert isinstance(mean, tuple) and isinstance(std, tuple)
        assert len(mean) == 3 and len(std) == 3
        self.mean = mean
        self.std = std

    def __call__(self, sample) -> dict:
        """Return transformed image and corresponding label"""
        image, label = sample["image"], sample["label"]
        # Transformation routine for normalization
        # 1. ToTensor = Takes PIL image (H, W, C) and transformed into tensor
        # with value between 0 <= val <= 1 (C, H, W)
        # 2. Normalize = Using PyTorch implementation, substract the mean
        # and divides by std
        trans = transforms.Compose([
            ToTensor(),
            Normalize(self.mean, self.std)
        ])
        img_trans = trans(image)
        # Transform label from PIL to tensor
        lbl_trans = ToTensor()(label)
        # @NOTE: uncomment following code if Label is not transformed
        # lbl_trans = label 
        return {"image":img_trans, "label":lbl_trans}


class Denormalize(object):
    """Denormalize image using PyTorch Normalization and pre-defined mean and std"""
    def __init__(self, 
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.255), 
        std=(1/0.229, 1/0.224, 1/0.255)) -> None:
        assert isinstance(mean, tuple) and isinstance(std, tuple)
        assert len(mean) == 3 and len(std) == 3
        self.mean = mean
        self.std = std

    def __call__(self, sample) -> dict:
        """Return denormalized transformed image and corresponding label"""
        image, label = sample["image"], sample["label"]
        # @NOTE: only denormalize image
        trans = Normalize(self.mean, self.std)
        img_trans = trans(image)
        return {"image":img_trans, "label":label}


def show_colored_label(sample):
    # convert sample to numpy array object
    sample = np.asarray(sample)
    colorized_label = Labels.colorize(sample)
    plt.imshow(colorized_label)


def test_script():
    # Create CustomDataset object
    dataset = CustomDataset()
    # show sampled image
    sample_index = 1
    print("image-0", dataset[1]["image"]) # 3 channel image (RGB)
    print("label-0", dataset[1]["label"]) # label only has one channel
    plt.imshow(dataset[sample_index]["image"])

    crop = CustomRandomCrop()
    flip = CustomRandomHorizontalFlip()
    norm = CustomNormalize()
    comb = transforms.Compose([
        crop,
        flip,
        norm
    ])
    # apply each of the transformation on selected sample
    sample_index = 1
    trans_sample = crop(dataset[sample_index])
    plt.figure()
    plt.imshow(trans_sample["image"])
    plt.figure()
    plt.imshow(trans_sample["label"])

    image = np.asarray(trans_sample["image"])
    label = np.asarray(trans_sample["label"])

    _, axs = plt.subplots(2, figsize=(15, 15))
    # show image and colorized label in one plot 
    axs[0].imshow(image)
    axs[0].set_title("Image Data")
    axs[1].imshow(Labels.colorize(label))
    axs[1].set_title("Colorized Label")

    trans_sample = flip(dataset[sample_index])
    plt.figure()
    plt.imshow(trans_sample["image"])
    plt.figure()
    plt.imshow(trans_sample["label"])

    trans_sample = comb(dataset[sample_index])
    plt.figure()
    # Display image from tensor to RGB
    plt.imshow(transforms.ToPILImage()(trans_sample["image"]).convert("RGB"))
    plt.figure()
    #plt.imshow(transforms.ToPILImage()(trans_sample["label"]).convert("RGB"))
    plt.imshow(trans_sample["label"])
    print(np.asarray(trans_sample["label"]))

    show_colored_label(trans_sample["label"])


if __name__ == "__main__":
    # Iterate over through the whole dataset, visualise image and colouful label, and save them to output directory!
    # Use `DataLoder` class from torch.utils.data module to load your `CustomDataset` class!
    #                   -> set batch_size to 1!
    
    # Define dataset transform routine
    crop = CustomRandomCrop()
    flip = CustomRandomHorizontalFlip()
    norm = CustomNormalize()
    comb = transforms.Compose([
        crop,
        flip,
        norm
    ])
    transformed_dataset = CustomDataset(transform=comb)
    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True)

    # iterate through dataloader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched["image"].size(), sample_batched["label"].size())
        transformed_dataset.shows(sample_batched["image"], sample_batched["label"])
# %%
