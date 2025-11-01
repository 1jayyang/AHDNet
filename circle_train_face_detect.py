import os
import shutil
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
import random
from pytorch_classification.MobileViT.model import *
from pytorch_classification.vision_transformer.vit_model import *

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, confusion_matrix
from loss import *
from tensorboardX import SummaryWriter
from face_data_dataloader import PairedFaceDataset, BalancedBatchSampler, BalancedBatchSampler_new_seed
from Image_Classification.models.resnet import *
from Image_Classification.models.alexnet import *
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
folder_12_batchsize = 2
num_epochs = 200
learning_rate = 1e-4
deterministic = True
random_seed = 1337

dataset_list = ['3', '4', '5', '6']
# network_list = ['resnet50', 'resnet101', 'resnet152','mobile_vit_small', 'mobile_vit_x_small', 'mobile_vit_xx_small', 'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224_in21k', 'vit_huge_patch14_224_in21k']
network_list = ['resnet34']

base_dataset_dir = "/"  # 替换为实际路径
base_save_path = "/"

if not deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

for network in network_list:
    for item in dataset_list:
        data_dir = os.path.join(base_dataset_dir, item + '_cropped')
        save_path = os.path.join(base_save_path, f"face_abnormal_detect_{network}_{item}")

        log_file = save_path + "/training_log.txt"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        with open(log_file, "w") as file:
            file.write("Epoch\tLoss\tRecall0\tRecall1\tF1\n")

        # 代码存储
        if os.path.exists(save_path + '/my_code'):
            shutil.rmtree(save_path + '/my_code')
        shutil.copytree('.', save_path + '/my_code',
                        shutil.ignore_patterns(['.git', '__pycache__']))

        # 加载训练集
        train_dir = os.path.join(data_dir, "train")
        train_dataset = ImageFolder(root=train_dir, transform=transform)
        # 获取类别索引
        folder_12_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 1]
        folder_0_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 0]
        # 自定义数据采样逻辑
        batch_sampler = BalancedBatchSampler_new_seed(folder_0_indices, folder_12_indices, batch_size, random_seed)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True)
        # "验证集数据加载"
        val_dir = os.path.join(data_dir, "val")
        val_dataset = ImageFolder(root=val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        if network == 'mobile_vit_small':
            model = mobile_vit_small(num_classes=2).to(device)
        elif network == 'mobile_vit_x_small':
            model = mobile_vit_x_small(num_classes=2).to(device)
        elif network == 'mobile_vit_xx_small':
            model = mobile_vit_xx_small(num_classes=2).to(device)
        elif network == 'vit_base_patch16_224':
            model = vit_base_patch16_224(num_classes=2).to(device)
        elif network == 'vit_base_patch32_224':
            model = vit_base_patch32_224(num_classes=2).to(device)
        elif network == 'vit_large_patch16_224':
            model = vit_large_patch16_224(num_classes=2).to(device)
        elif network == 'vit_large_patch32_224_in21k':
            model = vit_large_patch32_224_in21k(num_classes=2).to(device)
        elif network == 'vit_huge_patch14_224_in21k':
            model = vit_huge_patch14_224_in21k(num_classes=2).to(device)
        elif network == 'resnet18':
            model = resnet18(num_classes=2).to(device)
        elif network == 'resnet34':
            model = resnet34(num_classes=2).to(device)
        elif network == 'resnet50':
            model = resnet50(num_classes=2).to(device)
        elif network == 'resnet101':
            model = resnet101(num_classes=2).to(device)
        elif network == 'resnet152':
            model = resnet152(num_classes=2).to(device)

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 训练过程
        itr = 0
        best_performance = 0
        best_val_auc = 0
        best_f1 = 0
        iterator = tqdm(range(num_epochs), ncols=70)

        for epoch in iterator:
            epoch_seed = random_seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed(epoch_seed)

            batch_sampler = BalancedBatchSampler_new_seed(folder_0_indices, folder_12_indices, batch_size, epoch_seed)
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True)

            model.train()
            total_loss = 0
            correct = 0
            total = 0
            train_labels, train_preds = [], []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(preds)

                itr += 1
                print(f"Network:{network}, Dataset:{item}_cropped, iteration: {itr}-------- Loss: {loss:.4f}")

            tn, fp, fn, tp = confusion_matrix(train_labels, train_preds).ravel()

            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0

            f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
            f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

            f1_macro = (f1_0 + f1_1) / 2

            logging.info(
                f"Epoch {epoch}, Train_Loss: {total_loss:.4f}, Recall0: {recall_0:.4f}, Recall1: {recall_1:.4f}, Macro F1: {f1_macro:.4f}")
            with open(log_file, "a") as file:
                file.write(f"{epoch}\t{total_loss:.4f}\t{recall_0:.4f}\t{recall_1:.4f}\t{f1_macro:.4f}\n")

            if epoch > 0:
                model.eval()
                correct_val = 0
                total_val = 0
                val_labels, val_preds = [], []

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        preds = outputs.argmax(dim=1).detach().cpu().numpy()

                        val_labels.extend(labels.cpu().numpy())
                        val_preds.extend(preds)

                tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()

                recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

                precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
                precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0

                f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
                f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

                f1_macro = (f1_0 + f1_1) / 2

                logging.info(
                    f"Epoch {epoch}-----------Validation, Recall0: {recall_0:.4f}, Recall1: {recall_1:.4f}, Macro F1: {f1_macro:.4f}")
                with open(log_file, "a") as file:
                    file.write(f"'validation''----------'{recall_0:.4f}\t{recall_1:.4f}\t{f1_macro:.4f}\n")

                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    save_mode_path = os.path.join(save_path,
                                                  f'epoch_{epoch}_f1_{best_f1:.4f}.pth')
                    save_best = os.path.join(save_path,
                                             f'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                model.train()










