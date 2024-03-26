import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import sys

import torchattacks
from torchattacks import PGD

import numpy as np

from tqdm import tqdm


device = "cuda:1"


def train(model, train_loader, optimizer, criterion, adv_train=False, atk=None):
    model.train()
    total_loss = 0
    train_corrects = 0
    train_sum = 0
    for i, (image, label) in enumerate(tqdm(train_loader)):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        adv_image = atk(image, label)
        target = model(adv_image)

        loss = criterion(target, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        max_value, max_index = torch.max(target, 1)
        pred_label = max_index.cpu().numpy()
        true_label = label.cpu().numpy()
        train_corrects += np.sum(pred_label == true_label)
        train_sum += pred_label.shape[0]

        optimizer.zero_grad()
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


def evaluate(model, test_loader, criterion, adv_test=False, atk=None):
    model.eval()
    corrects = eval_loss = 0
    test_sum = 0
    for image, label in tqdm(test_loader):
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
    return eval_loss / float(len(test_loader)), corrects, corrects / test_sum


if __name__ == "__main__":
    print("start training")

    batch_size = 64
    learning_rate = 5e-5
    epoches = 50

    # 数据增强

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(20),  # 随机旋转角度
            transforms.ColorJitter(brightness=0.1),  # 颜色亮度
            transforms.Resize([224, 224]),  # 设置成224×224大小的张量
            transforms.ToTensor(),  # 将图⽚数据变为tensor格式
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),  # 将图⽚数据变为tensor格式
        ]
    )

    dataset_path = "face_dataset"
    train_path = dataset_path + "/train"
    val_path = dataset_path + "/test"
    train_data = datasets.ImageFolder(train_path, transform=train_transform)
    val_data = datasets.ImageFolder(val_path, transform=val_transform)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 2)
    # m_state_dict = torch.load("d4/epoch16.pt")
    # model.load_state_dict(m_state_dict)
    model = model.to(device)

    atk = PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoches):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, adv_train=True, atk=atk
        )
        print(
            "epoch"
            + str(epoch + 1)
            + "  train_loss:"
            + str(train_loss)
            + "  train_acc:"
            + str(train_acc)
        )
        if epoch % 5 == 0:
            test_loss, d, test_acc = evaluate(model, val_loader, criterion)
            print(
                "epoch"
                + str(epoch + 1)
                + "  val_loss:"
                + str(test_loss)
                + "  val_acc:"
                + str(test_acc)
            )
            test_loss, d, test_acc = evaluate(
                model, val_loader, criterion, adv_test=True, atk=atk
            )
            print(
                "epoch"
                + str(epoch + 1)
                + "  adv_val_loss:"
                + str(test_loss)
                + "  adv_val_acc:"
                + str(test_acc)
            )
            torch.save(model.state_dict(), "train2_epoch" + str(epoch + 1) + ".pt")
    test_loss, d, test_acc = evaluate(model, val_loader, criterion)
    print(
        "epoch"
        + str(epoch + 1)
        + "  val_loss:"
        + str(test_loss)
        + "  val_acc:"
        + str(test_acc)
    )
    test_loss, d, test_acc = evaluate(
        model, val_loader, criterion, adv_test=True, atk=atk
    )
    print(
        "epoch"
        + str(epoch + 1)
        + "  adv_val_loss:"
        + str(test_loss)
        + "  adv_val_acc:"
        + str(test_acc)
    )
    torch.save(model.state_dict(), "train2_epoch" + str(epoches) + ".pt")
    print("finish")
