import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import time

train_data=pd.read_csv(r'/myd2l/data/classify-leaves/train.csv')
test_data=pd.read_csv(r'/myd2l/data/classify-leaves/test.csv')


encoded_labels, unique_classes = pd.factorize(train_data['label'])
label_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}



# train_labels = torch.tensor(encoded_labels, dtype=torch.long)
# train_features= []
#
# # 图像尺寸调整大小为统一的大小，例如 (224, 224)
# image_size = (224, 224)
#
# # 定义图像预处理变换
# transform = Compose([
#     Resize(image_size),
#     ToTensor()
# ])
#
# for image_path in train_data['image']:
#     full_image_path=os.path.join('D:\\PycharmProjects\\learnpytorch\\myd2l\\data\\classify-leaves', image_path)
#
#     img= Image.open(full_image_path).convert('RGB')
#     img_tensor=transform(img)
#
#     train_features.append(img_tensor)
#
#
#
# train_features= torch.stack(train_features)
#
# # 保存预处理后的特征和标签
# torch.save(train_features, 'train_features.pt')
# torch.save(train_labels, 'train_labels.pt')
#
# print("Train features shape:", train_features.shape)  # 输出特征形状
# print("Train labels shape:", train_labels.shape)       # 输出标签形状


# test_features=[]
# for image_path in test_data['image']:
#     full_image_path=os.path.join('D:\\PycharmProjects\\learnpytorch\\myd2l\\data\\classify-leaves', image_path)
#
#     img= Image.open(full_image_path).convert('RGB')
#     img_tensor=transform(img)
#
#     test_features.append(img_tensor)
# test_features= torch.stack(test_features)
#
# torch.save(test_features, 'test_features.pt')


train_features=torch.load('train_features.pt')
train_labels=torch.load('train_labels.pt')
test_features=torch.load('test_features.pt')



# train_dataset=TensorDataset(train_features,train_labels)
#
# # 设置随机种子保证可重复性
# torch.manual_seed(42)
#
# # 计算划分比例
# total_size = len(train_dataset)
# test_size = int(total_size * 0.3)
# train_size = total_size - test_size
#
# train_subset, test_subset = torch.utils.data.random_split(
#     train_dataset, [train_size, test_size]
# )
#
# train_dataloader=DataLoader(train_subset,batch_size=64,shuffle=True)
#
# test_dataloader=DataLoader(test_subset,batch_size=64,shuffle=False)
#
#
class Residual(nn.Module):
    def __init__(self,input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,stride=strides,padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self,x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        z=x
        if self.conv3:
            z=self.conv3(x)
        y+=z
        return F.relu(y)

def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk




# def train_ch6(net,train_iter,test_iter,num_epochs,lr,device,writer=None):
#
#     #初始化部分
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#
#     if writer is None:
#         writer = SummaryWriter()
#
#     net.apply(init_weights)
#     print('training on', device)
#     net.to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr , weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
#     loss = nn.CrossEntropyLoss()
#
#     total_start = time.time()
#
#
#
#     #训练部分
#     for epoch in range(num_epochs):
#         epoch_start = time.time()
#
#         #训练
#         net.train()
#
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#
#         for x,y in train_iter:
#             x,y=x.to(device),y.to(device)
#             optimizer.zero_grad()
#             y_hat=net(x)
#             l=loss(y_hat,y)
#             l.backward()
#             optimizer.step()
#
#             train_l_sum+=l.item()*y.size(0)
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#             n+=y.size(0)
#
#         train_loss=train_l_sum/n
#         train_acc = train_acc_sum / n
#
#
#         #测试部分
#         test_l_sum, test_acc_sum, test_n = 0.0, 0.0, 0
#
#         net.eval()
#         with torch.no_grad():
#             for x,y in test_iter:
#                 x, y = x.to(device), y.to(device)
#                 y_hat=net(x)
#                 test_l_sum+=loss(y_hat,y).item()*y.size(0)
#                 test_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#                 test_n+=y.size(0)
#
#         test_loss=test_l_sum/test_n
#         test_acc = test_acc_sum / test_n
#         epoch_time = time.time() - epoch_start
#
#
#         # 记录指标
#         writer.add_scalars('accuracy',
#                            {'train_acc': train_acc, 'test_acc': test_acc},
#                            epoch)
#         writer.add_scalars('loss',
#                            {'train_loss': train_loss, 'test_loss': test_loss},
#                            epoch)
#
#         print(f'Epoch {epoch + 1}, '
#               f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.3f}, '
#               f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}, '
#               f'Time: {epoch_time:.1f}s')
#
#         scheduler.step(test_loss)  # 每个epoch结束后根据验证损失调整
#
#
#     # 计算总耗时
#     total_time = time.time() - total_start
#     h = int(total_time // 3600)
#     m = int((total_time % 3600) // 60)
#     s = total_time % 60
#     print(f'\nTotal training time: {h}h {m}m {s:.1f}s')
#
#     # 关闭writer
#     writer.close()
#
# b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#                    nn.BatchNorm2d(64), nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# b3 = nn.Sequential(*resnet_block(64, 128, 2))
# b4 = nn.Sequential(*resnet_block(128, 256, 2))
# b5 = nn.Sequential(*resnet_block(256, 512, 2))
#
# num_leaves=len(unique_classes)
#
# net = nn.Sequential(b1, b2, b3, b4, b5,
#                     nn.AdaptiveAvgPool2d((1,1)),
#                     nn.Flatten(), nn.Linear(512, num_leaves))
#
#
# lr=0.001
# num_epochs=20
# writer=SummaryWriter('./logs')
#
#
# train_ch6(net,train_dataloader,test_dataloader,num_epochs,lr,torch.device('cuda'),writer)
#
# torch.save(net, 'leaf_classifier.pth')


test_dataset=TensorDataset(test_features)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=False)

net=torch.load('leaf_classifier.pth')


# 预测函数
def predict(net, data_loader, device):
    net.eval()
    all_preds = []
    with torch.no_grad():
        for x in data_loader:
            x=x[0].to(device)
            outputs = net(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
pred_labels = predict(net, test_dataloader, device)

# 转换数字标签为实际类别名
pred_class_names = unique_classes[pred_labels]

# 生成提交文件
submission = pd.DataFrame({
    'image': test_data['image'],
    'label': pred_class_names
})

# 保存结果（路径与train.csv同级）
submission_path = r'/myd2l/submission/leaves_classify_submission.csv'
submission.to_csv(submission_path, index=False)