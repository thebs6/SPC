import argparse

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PDataSet import PDataSet
ImageFolder()

@torch.inference_mode()
def test_epoch(dataloder, model):
    accuracy = 0.0
    model.eval()
    pbar = tqdm(dataloder)
    pbar.set_postfix(accuracy=accuracy / (dataloder.batch_size * len(dataloder)))
    for batch, target in pbar:
        batch, target = batch.cuda(), target.cuda()
        output = model(batch)
        accuracy += (output.argmax(1) == target).sum()
    return accuracy / (dataloder.batch_size * len(dataloder))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder',
                        default=r'D:\Study\fly\paper_code\fgvc-aircraft-2013b.tar\fgvc-aircraft-2013b\data\images')
    parser.add_argument('--init_csv', default=r'D:\Study\fly\paper_code\SPC\all_df3.csv')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--t_batch', default=64, type=int)
    parser.add_argument('--v_batch', default=64, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--model_folder', default='./model')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--scheduler', default='StepLR')
    parser.add_argument('--step_size', default=15, type=int)
    parser.add_argument('--T_max', default=20)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--pretrain', default=False)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--freeze', default=True)
    parser.add_argument('--model_mode', default=0, type=int)
    parser.add_argument('--account', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    test_transform = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    args = parse_opt()
    test_data = PDataSet(args, 'all_df3.csv', 'test', test_transform)
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(test_data.classes))
    model = torch.load('test_models/model_58_0.7954.pth')
    model = model.cuda()
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
    test_accuracy = test_epoch(test_loader, model)

    print('test_accuracy:', test_accuracy)
