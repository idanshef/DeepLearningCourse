import os
import wget
import gzip
import shutil


base_dir = "/home/idansheffer/repos/others/DeepLearningCourse/HW1"

fashion_data_dir = os.path.join(base_dir, "data")
net_weights_dir = os.path.join(base_dir, "net_weights")

weight_decay_net_path = os.path.join(net_weights_dir, "WeightDecayLeNet5.pt")
batch_norm_net_path = os.path.join(net_weights_dir, "BatchNormLeNet5.pt")
dropout_net_path = os.path.join(net_weights_dir, "DropoutLeNet5.pt")


def _gunzip(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as f_in:
        with open(dest_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def _download_and_extract_file(local_zip_path, url):
    parent_dir = os.path.dirname(local_zip_path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)

    local_file_path = local_zip_path[:-3]
    if (not os.path.exists(local_file_path)) and (not os.path.exists(local_zip_path)):
        wget.download(url, local_zip_path)
    if os.path.exists(local_zip_path):
        _gunzip(local_zip_path, local_file_path)
        os.remove(local_zip_path)

    return local_file_path


def _download_data(local_dir, data_type):
    url_images = f"https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/{data_type}-images-idx3-ubyte.gz"
    url_labels = f"https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/{data_type}-labels-idx1-ubyte.gz"

    images_zip_path = os.path.join(local_dir, f"{data_type}-images-idx3-ubyte.gz")
    labels_zip_path = os.path.join(local_dir, f"{data_type}-labels-idx1-ubyte.gz")

    images_path = _download_and_extract_file(images_zip_path, url_images)
    labels_path = _download_and_extract_file(labels_zip_path, url_labels)

    return images_path, labels_path


def download_train_data(local_dir):
    return _download_data(local_dir, "train")


def download_test_data(local_dir):
    return _download_data(local_dir, "t10k")


