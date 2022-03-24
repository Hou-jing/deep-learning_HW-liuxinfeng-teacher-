# CNN处理图片
# transform
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
torch.cuda.empty_cache()
# train_trans = transforms.Compose([
#
#     transforms.Resize((128, 128)),  # 3*
#     transforms.ToTensor()
# ])
# test_trans = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

train_trans = test_trans = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.Resize((128, 128)),
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
def same_seeds(seed):
    # torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
same_seeds(0)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3,1,1)
        self.max_pool1 = nn.MaxPool2d(2,2,0)
        self.conv2 = nn.Conv2d(32, 64, 3,1,1)
        self.max_pool2 = nn.MaxPool2d(2,2,0)
        self.conv3= nn.Conv2d(64, 128, 3,1,1)
        self.fc1 = nn.Linear(128*32*32, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x=self.conv3(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
device='cuda' if torch.cuda.is_available() else 'cpu'
model=ConvNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS=15
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
        # Clip the gradient norms for stable training.梯度裁剪,梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # 最大范数是10
        optimizer.step()
        acc = (pre.ge(0.5).int() == labels).sum().item()
        train_loss.append(loss.item())
        train_acc.append(acc)
    train_loss = sum(train_loss) / len(train_loader)
    train_acc = sum(train_acc) / len(train_set)
    print(f'train|{epoch}/{EPOCHS},loss={train_loss:.3f},acc={train_acc:.3f}')
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
# def train(model, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data,target=data.to(device),target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.binary_cross_entropy(output, target)
#
#         loss.backward()
#         optimizer.step()
#
#
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device),target.to(device).float().reshape(50, 1)
#             output = model(data)
#             test_loss += F.binary_cross_entropy(output, target,reduction='sum').item()
#             pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in
#                                  output]).to(device)
