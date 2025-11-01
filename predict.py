import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from sklearn.metrics import recall_score
import torch.nn.functional as F
from Image_Classification.models.resnet import *


batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置数据的路径
save_path = "/"
data_dir = "/"  # 替换为实际路径
test_dir = os.path.join(data_dir, "test")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = resnet34(num_classes=2)
model.to(device)

best_model_path = os.path.join(save_path, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path))
model.eval()

correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        total += labels.size(0)

test_acc = 100 * correct / total

recall_class_0 = recall_score(all_labels, all_preds, labels=[0], average=None)[0]
recall_class_1 = recall_score(all_labels, all_preds, labels=[1], average=None)[0]

print(f"Test Recall Class 0 = {recall_class_0:.4f}, Test Recall Class 1 = {recall_class_1:.4f}, Test Accuracy: {test_acc:.2f}%")