import argparse
import os.path

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from PDataSet import PDataSet
import wandb
from torch import nn

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

api_key = '00f7f722b448e3badcd722a4d7d9f54a4493af68'
wandb.login(key=api_key)


def train_epoch(epoch, dataloader, model, loss_fn):
    model.eval()
    epoch_loss = 0.0
    accuracy = 0.0
    epoch_data_sum = 0
    bar = tqdm(dataloader, total=len(dataloader), position=0)
    for batch, target in bar:
        batch, target = batch.cuda(), target.cuda()
        output = model(batch)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        accuracy += (output.argmax(1) == target).sum()
        optimizer.step()
        epoch_loss += loss.item()
    return accuracy / (dataloader.batch_size * len(dataloader)), epoch_loss / len(dataloader)


@torch.inference_mode()
def valid_epoch(epoch, dataloader, model, loss_fn):
    epoch_loss = 0.0
    accuracy = 0.0
    model.eval()
    for batch, target in tqdm(dataloader, position=0):
        batch, target = batch.cuda(), target.cuda()
        output = model(batch)
        loss = loss_fn(output, target)
        accuracy += (output.argmax(1) == target).sum()
        epoch_loss += loss.item()
    return accuracy / (dataloader.batch_size * len(dataloader)), epoch_loss / len(dataloader)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder',
                        default=r'D:\Study\fly\experiments\crowl_ps_using\images_origin_remove_duplicate')
    parser.add_argument('--init_csv', default='init2000.csv')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--t_batch', default=64, type=int)
    parser.add_argument('--v_batch', default=64, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--model_folder', default='./model')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--scheduler', default='StepLR')
    parser.add_argument('--step_size', default=15, type=int)
    parser.add_argument('--T_max', default=20)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--pretrain', default=False)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--freeze', default=True)
    parser.add_argument('--model_mode', default=0, type=int)
    parser.add_argument('--account', type=str)
    parser.add_argument('--image_folder_type', default=1, type=int)


    args = parser.parse_args()
    return args


def get_scheduler(optimizer, type, step_size, T_max):
    scheduler = None
    if type not in ['StepLR', 'Cosine']:
        raise (ValueError, 'only support StepLR, Cosine')
    if type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif type == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20,
                                                               eta_min=0.00001)
    return scheduler


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


if __name__ == '__main__':
    args = parse_opt()
    wandb.init(project='SPC', config=vars(args))
    wandb.run.name = f"finetune_epoch:{args.epoch}_trainb_{args.t_batch}_validb_{args.v_batch}_lr_{args.lr}_scheduler_{args.scheduler}+_{args.account}"
    train_transform = tf.Compose([
        tf.RandomResizedCrop(args.image_size),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transform = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = PDataSet(args, args.init_csv, 'train', transform=train_transform)
    valid_data = PDataSet(args, args.init_csv, 'valid', transform=valid_transform)

    learn_rate = args.lr
    epoch_num = args.epoch
    train_batch = args.t_batch
    valid_batch = args.v_batch
    min_loss = 5
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(args.pretrain)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(args.pretrain)
    if args.freeze:
        model = freeze_model(model)

    if args.model_mode == 0:
        model.fc = torch.nn.Linear(model.fc.in_features, len(train_data.classes))
    elif args.model_mode == 1:
        model.fc = torch.nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(train_data.classes)),
            nn.LogSoftmax(dim=1)
        )
    model = model.cuda()

    if args.finetune_mode == 0:
        for parameters in model.parameters():
            parameters.requires_grad = False

        for parameters in model.fc.parameters():
            parameters.requires_grad = True


    loss_fn = CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = get_scheduler(optimizer, type=args.scheduler, step_size=args.step_size, T_max=args.T_max)

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True, num_workers=args.workers,
                              drop_last=False)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=valid_batch, shuffle=True, num_workers=args.workers,
                              drop_last=False)

    for epoch in tqdm(range(1, epoch_num + 1), position=0):
        if args.freeze and epoch == epoch_num // 2:
            model = unfreeze_model(model)
        train_accuracy, epoch_loss = train_epoch(epoch, train_loader, model, loss_fn)
        scheduler.step()
        valid_accuracy, valid_loss = valid_epoch(epoch, valid_loader, model, loss_fn)
        wandb.log(
            {
                'train_loss': epoch_loss,
                'train_accuracy': train_accuracy,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
                'lr': optimizer.state_dict()['param_groups'][0]['lr']
            }
        )

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            if not os.path.exists(args.model_folder):
                os.mkdir(args.model_folder)
            torch.save(model, f'{args.model_folder}/model_' + str(epoch) + ".pth")
