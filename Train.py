from PoseDataset import Posedataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from DSACNet import DSACNet

def adjust_learning_rate(optimizer, gamma, step):
    lr = LR * (gamma ** (step))
    print('change learning rate, now learning rate is :', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume = None
    Epoch = 5
    Batch_size = 16
    lr_reduce_epoch = 15
    model_save_epoch = 10
    gamma = 0.2
    LR = 0.0001
    pic_resize = (128, 128) # 调整图片大小,要符合网络输入大小
    weight_path = r'D:\PycharmProjects\PoseEstimate\Weights'
    resume_path = os.path.join(weight_path, 'ELNetV2_on_stimulatedata-loss-379.77.pth')
    anno_path = r'D:\PycharmProjects\PoseEstimate\data_real\anno_train.txt'
    img_path = r'D:\PycharmProjects\PoseEstimate\data_real\img_together'

    transform = transforms.Compose([
        transforms.Resize(pic_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    train_data = Posedataset(anno_path=anno_path, img_path=img_path, transform=transform)
    train_data_loader = DataLoader(train_data, batch_size=Batch_size, shuffle=True)

    netname = 'DSACNet'
    net = DSACNet().to(device)

    if resume:
        # 接着训练
        print('loading checkpoint...')
        net.load_state_dict(torch.load(resume_path))

    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    plotloss = []
    plotauc = []

    print('start training........')
    train_start_time = time.time()
    for epoch in range(Epoch):
        if epoch % lr_reduce_epoch == 0:
            adjust_learning_rate(optimizer, gamma, epoch//lr_reduce_epoch)

        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (images, labels) in enumerate(train_data_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 更新测试图片的数量
            correct += (predicted == labels).sum()  # 更新正确分类的图片的数量
            if i % 20 == 0:
                print('[epoch:%d, batch_idx:%d] Loss: %.03f | Acc: %.3f%% ' % (epoch+1, i, sum_loss/(i+1), (correct.cpu().numpy() / total)*100))

            plotloss.append(sum_loss / (i + 1))
            plotauc.append((correct.cpu().numpy() / total)*100)

        if (epoch+1) % model_save_epoch == 0:
            torch.save(net.state_dict(), os.path.join(weight_path, netname + '-loss-{:.2f}.pth'.format(sum_loss)))
            print('moodel save success!')

    print('Finished Training')
    train_end_time = time.time()
    print('The total training time:{}'.format(train_end_time-train_start_time))

    # # #存贮训练精度/损失数据
    # loss_file_path = r'D:\PycharmProjects\PoseEstimate\Weights\ELNetV2\transferlearning_record\record'+'/' + netname + '_' + 'loss' + '.txt'
    # acc_file_path = r'D:\PycharmProjects\PoseEstimate\Weights\ELNetV2\transferlearning_record\record'+'/' + netname + '_' + 'acc' + '.txt'
    # file_loss = open(loss_file_path, "w")
    # file_acc = open(acc_file_path, "w")
    # file_loss.write(str(plotloss))
    # file_acc.write(str(plotauc))

    #绘制训练精度/损失图
    plt.subplot(2,1,1)
    plt.plot(plotloss)
    plt.xlabel('Batch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.subplot(2,1,2)
    plt.plot(plotauc)
    plt.xlabel('Batch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.tight_layout()
    plt.show()