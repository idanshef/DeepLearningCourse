import os
import wget
import gzip
import shutil


url_train_images = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
url_train_labels = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
url_test_images = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
url_test_labels = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"


def gunzip(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as f_in:
        with open(dest_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_and_extract_file(local_path, url):
    parent_dir = os.path.dirname(local_path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)

    if (not os.path.exists(local_path[:-3])) and (not os.path.exists(local_path)):
        wget.download(url, local_path)
    if os.path.exists(local_path):
        gunzip(local_path, local_path[:-3])
        os.remove(local_path)


def download_train_data(local_dir):
    train_images_path = os.path.join(local_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(local_dir, "train-labels-idx1-ubyte.gz")

    download_and_extract_file(train_images_path, url_train_images)
    download_and_extract_file(train_labels_path, url_train_labels)


def download_test_data(local_dir):
    test_images_path = os.path.join(local_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(local_dir, "t10k-labels-idx1-ubyte.gz")

    download_and_extract_file(test_images_path, url_test_images)
    download_and_extract_file(test_labels_path, url_test_labels)

