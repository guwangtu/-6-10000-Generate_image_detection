import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import os
import sys

import torchattacks
from torchattacks import PGD

import numpy as np
import cv2

from tqdm import tqdm

from scripts.argument import parser
import logging

from scripts.load_data import load_artifact, load_fold, load_diffusion_forensics, load_GenImage


class Trainer:
    def __init__(self, args, atk):
        self.args = args
        self.atk = atk
        self.device = "cuda:" + args.device
        self.loggers = []
        self.train_loader = None
        self.train_loader2 = None
        self.val_loader = None
        self.val_loader2 = None
        self.epoch = args.load_epoch

        save_path = "checkpoint/" + args.save_path
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        int_files = [int(file) for file in os.listdir(save_path)]
        if len(int_files) == 0:
            save_path = os.path.join(save_path, "1")
        else:
            save_path = os.path.join(save_path, str(max(int_files) + 1))
        self.save_path = save_path  # 例：checkpoint/face_normal/3
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    def set_dataloader(
        self, train_loader=None, train_loader2=None, val_loader=None, val_loader2=None
    ):
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.val_loader = val_loader
        self.val_loader2 = val_loader2

    def train(
        self,
        model,
        adv_train=False,
    ):
        args = self.args

        criterion = torch.nn.CrossEntropyLoss()
        if args.sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        self.set_loggers(
            self.save_path + "/" + args.save_path
        )  # 例：...savepath/2/savepath          顺序train,test,advtest

        if args.test_first:
            self.evaluate(
                model,
                adv_test=args.adv or args.adv_test,
            )
        for epoch in range(args.load_epoch, args.epoches):
            self.epoch = epoch
            train_loss, train_acc, losses = self.train_step(
                model,
                optimizer,
                criterion,
                adv_train=adv_train,
            )
            print(
                "epoch"
                + str(epoch + 1)
                + "  train_loss:"
                + str(train_loss)
                + "  train_acc:"
                + str(train_acc)
            )
            self.loggers[0].info(f"Epoch{epoch}: Training accuracy: {train_acc:.4f}")
            np.save(
                self.save_path + "/batch_losse_epoch" + str(epoch) + ".npy",
                np.array(losses),
            )

            if (epoch + 1) % args.save_each_epoch == 0:
                self.evaluate(
                    model,
                    adv_test=args.adv_test or args.adv,
                )
                torch.save(
                    model.state_dict(), self.save_path + "/epoch" + str(epoch) + ".pt"
                )
            if args.adv and (epoch + 1) % args.update_adv_each_epoch == 0:
                self.atk = PGD(
                    model,
                    eps=args.atk_eps,
                    alpha=args.atk_alpha,
                    steps=args.atk_steps,
                    random_start=True,
                )
                self.atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
                self.atk.set_device(self.device)
        torch.save(
            model.state_dict(),
            self.save_path + "/final_epoch" + str(args.epoches) + ".pt",
        )
        print(f"Save in: {self.save_path}")

    def train_step(
        self,
        model,
        optimizer,
        criterion,
        adv_train=False,
    ):
        args = self.args
        atk = self.atk
        device = self.device

        model.train()

        total_loss = 0
        train_corrects = 0
        train_sum = 0

        if self.train_loader2:
            dataloader_iterator = iter(self.train_loader2)

        batch_count = 0
        losses = []
        for i, (image, label) in tqdm(enumerate(self.train_loader)):
            image = image.to(device)
            label = label.to(device)
            if self.train_loader2:
                try:
                    image2, label2 = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.train_loader2)
                    image2, label2 = next(dataloader_iterator)
                image2 = image2.to(device)
                label2 = label2.to(device)

                image = torch.cat([image, image2], dim=0)
                label = torch.cat([label, label2], dim=0)

            optimizer.zero_grad()
            if adv_train:
                adv_image = atk(image, label)
                img = torch.cat((image, adv_image), dim=0)
                label = torch.cat((label, label), dim=0)
                target = model(img)
            else:
                target = model(image)
            loss = criterion(target, label)

            loss.backward()
            optimizer.step()
            batch_count += 1

            total_loss += loss.item()
            max_value, max_index = torch.max(target, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            train_corrects += np.sum(pred_label == true_label)
            train_sum += pred_label.shape[0]
            losses.append(loss.item())
            if not args.test_each_batch == 0:
                if (i + 1) % args.test_each_batch == 0:
                    this_acc = np.sum(pred_label == true_label) / pred_label.shape[0]
                    test_loss, d, test_acc = self.evaluate_step(
                        model, self.val_loader, criterion, adv_test=False
                    )
                    self.loggers[0].info(
                        f"               Batch_id:{i} Batch Loss:{loss.item()} This acc: {this_acc} Normal Evaluate accuracy: {test_acc:.4f}"
                    )
                    if args.adv or args.adv_test:
                        test_loss, d, test_acc = self.evaluate_step(
                            model, self.val_loader, criterion, adv_test=True
                        )
                        self.loggers[0].info(
                            f"               Batch_id:{i} Batch Loss:{loss.item()} Adv Evaluate accuracy: {test_acc:.4f}"
                        )
                    model.train()
            # np.save("batch_losses.npy",np.array(losses))
        return (
            total_loss / float(len(self.train_loader)),
            train_corrects / train_sum,
            losses,
        )

    def evaluate(self, model, adv_test=False):
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, d, test_acc = self.evaluate_step(
            model, self.val_loader, criterion, adv_test=False
        )
        print("val_loss:" + str(test_loss) + "  val_acc:" + str(test_acc))
        self.loggers[1].info(
            f"Epoch{self.epoch}: Loss:{test_loss} Evaluate accuracy: {test_acc:.4f}"
        )
        if adv_test:
            test_loss, d, test_acc = self.evaluate_step(
                model, self.val_loader, criterion, adv_test=True
            )
            print("adv_val_loss:" + str(test_loss) + "  adv_val_acc:" + str(test_acc))
            self.loggers[2].info(
                f"Epoch{self.epoch}: Adv Loss:{test_loss} Adv evaluate accuracy: {test_acc:.4f}"
            )
        if self.val_loader2:
            test_loss, d, test_acc = self.evaluate_step(
                model, self.val_loader2, criterion, adv_test=False
            )
            print(
                "another_val_loss:"
                + str(test_loss)
                + "  another_val_acc:"
                + str(test_acc)
            )
            self.loggers[1].info(
                f"Epoch{self.epoch}: Loss:{test_loss} Another Evaluate accuracy: {test_acc:.4f}"
            )

    def evaluate_step(self, model, val_loader, criterion, adv_test=False):
        device = self.device
        atk = self.atk
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
                test_sum += pred_label.shape[0]
        return eval_loss / float(len(val_loader)), corrects, corrects / test_sum

    def set_loggers(self, save_path):
        args = self.args
        self.loggers = []

        train_logger = self.get_logger(save_path, "train")
        test_logger = self.get_logger(save_path, "test")
        self.loggers.append(train_logger)
        self.loggers.append(test_logger)
        if args.adv:
            adv_test_logger = self.get_logger(save_path, "adv_test")
            self.loggers.append(adv_test_logger)

    def get_logger(self, save_path, typestr):
        # 创建train和test日志记录器
        this_logger = logging.getLogger(typestr)
        this_logger.setLevel(logging.INFO)

        this_file_handler = logging.FileHandler(save_path + "_" + typestr + ".log")
        this_file_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        this_file_handler.setFormatter(formatter)

        # 将处理器添加到日志记录器
        this_logger.addHandler(this_file_handler)
        return this_logger

    def get_adv_imgs(self, data_loader):
        device = self.device
        args = self.args
        save_path = args.save_path
        atk = self.atk
        i = 0
        j = 0
        for image, label in tqdm(data_loader):
            image = image.to(device)
            label = label.to(device)
            imgs = atk(image, label)

            for t in range(len(label)):
                this_label = label[t].cpu().numpy().astype(np.uint8)
                os.makedirs(f"{save_path}/{str(this_label)}", exist_ok=True)
                if this_label == 0:
                    i += 1
                    k = i
                else:
                    j += 1
                    k = j
                torchvision.utils.save_image(
                    imgs[t], f"{save_path}/{str(this_label)}/{str(k)}.png"
                )
