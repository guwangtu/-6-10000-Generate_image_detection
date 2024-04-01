import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import os
import sys

import torchattacks
from torchattacks import PGD

import numpy as np

from tqdm import tqdm

from argument import parser


class Trainer:
    def __init__(self, args, atk):
        self.args = args
        self.atk = atk
        self.device = "cuda:" + args.device

    def train(self, model, train_loader, val_loader, adv_train=False):
        args = self.args
        atk = self.atk
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        save_path = args.save_path
        int_files = [int(file) for file in os.listdir(save_path)]
        if len(int_files) == 0:
            save_path = os.path.join(save_path, "1")
        else:
            save_path = os.path.join(save_path, str(max(int_files) + 1))
        
        os.mkdir(save_path)

        for epoch in range(args.load_epoch, args.epoches):
            train_loss, train_acc = self.train_step(
                model, train_loader, optimizer, criterion
            )
            print(
                "epoch"
                + str(epoch + 1)
                + "  train_loss:"
                + str(train_loss)
                + "  train_acc:"
                + str(train_acc)
            )
            if adv_train:
                train_loss, train_acc = self.train_step(
                    model, train_loader, optimizer, criterion, adv_train=True
                )
                print(
                    "epoch"
                    + str(epoch + 1)
                    + "  adv_train_loss:"
                    + str(train_loss)
                    + "  adv_train_acc:"
                    + str(train_acc)
                )

            if (epoch + 1) % args.save_each_epoch == 0:
                self.evaluate(model, val_loader, criterion, adv_test=True, atk=atk)
                torch.save(
                    model.state_dict(), save_path + "/epoch" + str(epoch + 1) + ".pt"
                )
        self.evaluate(model, val_loader, criterion, adv_test=True, atk=atk)
        torch.save(
            model.state_dict(), save_path + "/final_epoch" + str(args.epoches) + ".pt"
        )

    def train_step(self, model, train_loader, optimizer, criterion, adv_train=False):
        args = self.args
        atk = self.atk
        device = self.device

        model.train()

        total_loss = 0
        train_corrects = 0
        train_sum = 0

        for i, (image, label) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            if adv_train:
                adv_image = atk(image, label)
                target = model(adv_image)
            else:
                target = model(image)
            loss = criterion(target, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            max_value, max_index = torch.max(target, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            train_corrects += np.sum(pred_label == true_label)
            train_sum += pred_label.shape[0]

        return total_loss / float(len(train_loader)), train_corrects / train_sum

    def evaluate(self, model, val_loader, criterion, adv_test=False, atk=None):
        test_loss, d, test_acc = self.evaluate_step(model, val_loader, criterion)
        print("val_loss:" + str(test_loss) + "  val_acc:" + str(test_acc))
        if adv_test:
            test_loss, d, test_acc = self.evaluate_step(
                model, val_loader, criterion, adv_test=True, atk=atk
            )
            print("adv_val_loss:" + str(test_loss) + "  adv_val_acc:" + str(test_acc))

    def evaluate_step(self, model, val_loader, criterion, adv_test=False, atk=None):
        device = self.device

        model.eval()
        corrects = eval_loss = 0
        test_sum = 0
        for image, label in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)
            if adv_test:
                image = atk(image, label)
            with torch.no_grad():
                pred = model(image)
                loss = criterion(pred, label)
                eval_loss += loss.item()
                max_value, max_index = torch.max(pred, 1)
                pred_label = max_index.cpu().numpy()
                true_label = label.cpu().numpy()
                corrects += np.sum(pred_label == true_label)
                test_sum += np.sum(pred_label == true_label) + np.sum(
                    pred_label != true_label
                )
        return eval_loss / float(len(val_loader)), corrects, corrects / test_sum


def main(args):

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 2)

    batch_size = args.batch_size
    device = "cuda:" + str(args.device)

    if args.load_path:
        load_path = args.load_path
        m_state_dict = torch.load(load_path)
        model.load_state_dict(m_state_dict)

    model = model.to(device)

    atk = PGD(
        model,
        eps=args.atk_eps,
        alpha=args.atk_alpha,
        steps=args.atk_steps,
        random_start=True,
    )
    atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])

    trainer = Trainer(args, atk)

    if args.todo == "train":

        dataset_path = args.dataset
        train_path = dataset_path + "/train"
        val_path = dataset_path + "/test"
        save_path = args.save_path
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(20),  # 随机旋转角度
                transforms.ColorJitter(brightness=0.1),  # 颜色亮度
                transforms.Resize([224, 224]),  # 设置成224×224大小的张量
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                # std=[0.229, 0.224, 0.225]),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )

        train_data = datasets.ImageFolder(train_path, transform=train_transform)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_data = datasets.ImageFolder(val_path, transform=val_transform)
        val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

        trainer.train(model, train_loader, val_loader, args.adv_train)

    elif args.todo == "test":

        dataset_path = args.dataset
        val_path = dataset_path + "/test"
        val_data = datasets.ImageFolder(val_path, transform=val_transform)
        val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

        trainer.evaluate()


if __name__ == "__main__":
    
    args = parser()

    main(args)
