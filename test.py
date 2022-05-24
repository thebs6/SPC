import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

@torch.inference_mode()
def test_epoch(dataloder, model):
    accuracy = 0.0
    model.eval()
    for batch, target in tqdm(dataloder):
        batch, target = batch.cuda(), target.cuda()
        output = model(batch)
        accuracy += (output.argmax(1) == target).sum()
    return accuracy / (dataloder.batch_size * len(dataloder))


if __name__ == '__main__':
    test_transform = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = ImageFolder(r'D:\Study\fly\experiments\crowl_ps_using/test', transform=test_transform)
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(test_data.classes))
    model = torch.load('model_60.pth')
    model = model.cuda()
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    test_accuracy = test_epoch(test_loader, model)

    print('test_accuracy:', test_accuracy)
