import os
import shutil
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PATH_DATASET = os.path.abspath("../dataset")
ORIGINAL_PATH_DATASET_IMAGES = os.path.join(PATH_DATASET, "car_data/car_data")


def get_dataset_transformation(
    image_width: int = 512,
    image_height: int = 512,
    rotation: int = 15,
    normalization: int = 0.5,
) -> Tuple[transforms.Compose, transforms.Compose]:
    n = (normalization, normalization, normalization)
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_width, image_height)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation),
            transforms.ToTensor(),
            transforms.Normalize(n, n),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.Resize((image_width, image_height)),
            transforms.ToTensor(),
            transforms.Normalize(n, n),
        ]
    )
    return train_tfms, test_tfms


def create_subset(
    src_path_dataset: str,
    dst_path_dataset: str,
    included_directories: Optional[List[str]] = None,
) -> None:
    if os.path.exists(dst_path_dataset):
        shutil.rmtree(dst_path_dataset)

    if not included_directories:
        included_directories = os.listdir(os.path.join(src_path_dataset, "train"))

    def link(category: str) -> None:

        for included_directory in included_directories:

            original_dir_path = os.path.join(
                src_path_dataset, category, included_directory
            )
            new_dir_path = os.path.join(dst_path_dataset, category, included_directory)

            os.makedirs(new_dir_path)

            for file in os.listdir(original_dir_path):
                src = os.path.join(original_dir_path, file)
                dst = os.path.join(new_dir_path, file)
                os.link(src, dst)

    link("train")
    link("test")


def get_dataset_loader(
    path_dataset: str,
    train_tfms: transforms.Compose,
    test_tfms: transforms.Compose,
    batch_size: int = 32,
) -> Tuple[int, datasets.ImageFolder, datasets.ImageFolder]:
    train_dataset = datasets.ImageFolder(
        root=os.path.join(path_dataset, "train"), transform=train_tfms
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(path_dataset, "test"), transform=test_tfms
    )

    print(f"Number of classes {len(train_dataset.classes)}")
    print(f"Train dataset size is {len(train_dataset)}")
    print(f"Test dataset size is {len(test_dataset)}")

    # Convertion des datasets en dataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return len(train_dataset.classes), train_loader, test_loader


def create_unbalenced_dataset(
    src_path_dataset: str, dst_path_dataset: str, test_size: float
) -> None:
    if os.path.exists(dst_path_dataset):
        shutil.rmtree(dst_path_dataset)

    categories = os.listdir(os.path.join(src_path_dataset, "train"))
    paths_by_categories: Dict[str, List[str]] = {
        category: [] for category in categories
    }

    for var in ["train", "test"]:
        for category in categories:
            dir = os.path.join(src_path_dataset, var, category)
            paths = [os.path.join(dir, file) for file in os.listdir(dir)]
            paths_by_categories[category].extend(paths)

    paths_train_test_by_categories: Dict[str, Tuple[List[str], List[str]]] = {}
    for category, paths in paths_by_categories.items():
        train, test = train_test_split(paths, test_size=test_size)
        paths_train_test_by_categories[category] = (train, test)

    for category, train_test in paths_train_test_by_categories.items():
        train, test = train_test

        train_path = os.path.join(dst_path_dataset, "train", category)
        test_path = os.path.join(dst_path_dataset, "test", category)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for path in train:
            try:
                os.link(path, os.path.join(train_path, path.split("/")[-1]))
            except:
                continue

        for path in test:
            try:
                os.link(path, os.path.join(test_path, path.split("/")[-1]))
            except:
                continue


if __name__ == "__main__":
    # create a dataset 'dataset/car_data_80_20' (train 80%, test 20%)
    print("Dataset 'dataset/car_data_80_20/car_data' created")
    create_unbalenced_dataset(
        "../dataset/car_data/car_data", "../dataset/car_data_80_20/car_data", 0.2
    )