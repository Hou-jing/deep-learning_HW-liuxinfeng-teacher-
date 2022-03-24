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

train_trans = test_trans = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6272, 512)#128*7*7
        torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        torch.nn.Dropout(0.5)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
model = ConvNet()


device='cuda' if torch.cuda.is_available() else 'cpu'
# model=ConvNet().to(device)
model=model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


EPOCHS=20
for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    train_acc = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs,labels=imgs.to(device),labels.to(device)
        pre = model(imgs)
        labels=labels.unsqueeze(-1).float()
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
    print(f'valid|{epoch },validloss={valid_loss:.3f},validacc={valid_acc:.3f}')

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
