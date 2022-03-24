import torch
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from torch.nn import Flatten
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F

# train_trans = transforms.Compose([
#     # transforms.RandomResizedCrop(150),
#     transforms.FiveCrop(128),  # TypeError: pic should be PIL Image or ndarray. Got <class 'tuple'>
#     transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
# test_trans = transforms.Compose([
#     transforms.RandomResizedCrop(128),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
# train_trans=transforms.Compose([
#
#     transforms.RandomAffine(15,translate=None,scale=None,shear=None,resample=False,fillcolor=0),
#     transforms.Resize((128,128)),#3*
#     transforms.ToTensor()
# ])

train_trans=transforms.Compose([
                                   transforms.RandomAffine(degrees = 0,translate=(0.1, 0.1)),
                                   transforms.RandomRotation((-10,10)),#将图片随机旋转（-10,10）度,
                                    transforms.Resize((128,128)),
                                   transforms.ToTensor(),# 将PIL图片或者numpy.ndarray转成Tensor类型
                                   transforms.Normalize((0.1307,), (0.3081,))])
test_trans=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
train_set = DatasetFolder('D:\\pythonProject_hj_2\\猫狗分类2\\dogs_and_cats_small_2\\train',
                          loader=lambda x: Image.open(x), extensions='jpg', transform=train_trans)
test_set = DatasetFolder('D:\\pythonProject_hj_2\\猫狗分类2\\dogs_and_cats_small_2\\test', loader=lambda x: Image.open(x),
                         extensions='jpg', transform=test_trans)
valid_set = DatasetFolder('D:\\pythonProject_hj_2\\猫狗分类2\\dogs_and_cats_small_2\\validation',
                          loader=lambda x: Image.open(x), extensions='jpg', transform=train_trans)

batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True, pin_memory=True)


# model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_drop = nn.Dropout(0.25)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_drop = nn.Dropout(0.25)

        # Fully-Connected layer 1

        self.fc1 = nn.Linear(64*28*28, 256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)

        out = self.maxpool1(out)
        out = self.conv1_drop(out)

        # conv layer 2 的前向计算，4行代码
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)

        out = self.maxpool2(out)
        out = self.conv2_drop(out)

        # Flatten拉平操作
        out = out.view(out.size(0), -1)

        # FC layer的前向计算（2行代码）
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)

        out = self.fc2(out)
        out=torch.sigmoid(out)

        return out


# model = ConvNet()

model=CNNModel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model=ConvNet().to(device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    train_acc = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        pre = model(imgs)
        labels = labels.unsqueeze(-1).float()
        loss = F.binary_cross_entropy(pre, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (pre.ge(0.5).int() == labels).sum().item()
        train_loss.append(loss.item())
        train_acc.append(acc)
    train_loss = sum(train_loss) / len(train_loader)
    train_acc = sum(train_acc) / len(train_set)
    print(f'train|{epoch}/{EPOCHS},loss={train_loss:.4f},acc={train_acc:.3f}')
    valid_loss = []
    valid_acc = []
    model.eval()
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.unsqueeze(-1).float()
        with torch.no_grad():
            pre = model(imgs.to(device))
        acc = (pre.ge(0.5).int() == labels).sum().item()
        loss = F.binary_cross_entropy(pre, labels.to(device))
        valid_loss.append(loss.item())
        valid_acc.append(acc)
    valid_acc = sum(valid_acc) / len(valid_set)
    valid_loss = sum(valid_loss) / len(valid_loss)
    print(f'valid|{epoch},validloss={valid_loss:.3f},validacc={valid_acc:.3f}')

test_loss = []
test_acc = []
model.eval()
for batch in tqdm(test_loader):
    imgs, labels = batch
    imgs, labels = imgs.to(device), labels.to(device)
    labels = labels.unsqueeze(-1).float()
    with torch.no_grad():
        pre = model(imgs.to(device))
    acc = (pre.ge(0.5).int() == labels).sum().item()
    loss = F.binary_cross_entropy(pre, labels.to(device))
    test_loss.append(loss.item())
    test_acc.append(acc)
test_acc = sum(test_acc) / len(test_set)
test_loss = sum(test_loss) / len(test_loss)
print(f'test|,testloss={test_loss:.3f},testacc={test_acc:.3f}')
