import torch
import idx2numpy
from utils import *
from lenet5 import LeNet5


def predict_data(images_to_predict, gt_labels):
    net_weights_path = weight_decay_net_path
    net = LeNet5()
    net.to(device=device)
    net.load_state_dict(torch.load(net_weights_path, map_location=device))
    net.eval()

    with torch.no_grad():
        tensor_images = torch.from_numpy(images_to_predict).to(device=device, dtype=torch.float).unsqueeze(1)
        tensor_labels = torch.from_numpy(gt_labels).to(device=device, dtype=torch.long)

        pred_label_tensor = net(tensor_images)
        pred_label_arg = torch.argmax(pred_label_tensor, dim=1)

        correct_pred = int(torch.sum(pred_label_arg == tensor_labels))
        total = int(tensor_labels.size()[0])
        print(f"Accuracy: {correct_pred / total}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_path, labels_path = download_test_data(fashion_data_dir)
    images = idx2numpy.convert_from_file(images_path)
    labels = idx2numpy.convert_from_file(labels_path)

    predict_data(images, labels)
