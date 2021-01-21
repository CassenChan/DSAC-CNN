from PoseDataset import Posedataset
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from DSACNet import DSACNet

if __name__ == '__main__':

    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anno_path = r'D:\PycharmProjects\PoseEstimate\data_real\anno_test.txt'
    img_path = r'D:\PycharmProjects\PoseEstimate\data_real\img_together'
    weight_path = r'D:\PycharmProjects\PoseEstimate\Weights'
    Batch_size = 32
    pic_resize = (128, 128)
    classes = ['左偏67°', '左偏45°', '左偏22°', '正视00°', '右偏22°', '右偏45°', '右偏67°']

    transform = transforms.Compose([
        transforms.Resize(pic_resize), # 调整图片大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    test_data = Posedataset(anno_path=anno_path, img_path=img_path, transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=Batch_size, shuffle=True)

    net = DSACNet().to(device)
    net.load_state_dict(torch.load(weight_path))  # 加载模型
    net.eval()

    class_correct = list(0. for i in range(len(classes)))  #每一类数据预测正确的总数
    class_total = list(0. for i in range(len(classes)))  #每一类数据的总数
    total_acc = 0  #预测正确数据总数
    timelist = [] #存储每一batch的时间
    with torch.no_grad():
        prev_time = time.time()
        for data in test_data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            cond = (predicted == labels).squeeze()
            total_acc += (predicted == labels).sum().item()
            cond = cond.cpu().numpy()
            for i in range(len(labels)):
                # 对各个类的进行各自累加
                label = labels[i]
                label = label.cpu().numpy()
                class_correct[label] += cond[i]
                class_total[label] += 1

            current_time = time.time()
            batch_time = current_time-prev_time
            timelist.append(batch_time)
            prev_time = current_time
        # timelist = timelist[1:] #第一次涉及到系统资源分配等时间，所以剔除
        ave_time = sum(timelist) / len(timelist)
        print('Average batch time : {:.5f}s'.format(ave_time))
        print('One picture time : {:.5f}s'.format(ave_time/Batch_size))

        for cls in range(len(classes)):
            accclass = 100 * (class_correct[cls] / class_total[cls])
            print('Accuracy of {} : {:.3f}%'.format(classes[cls], accclass))

        total_accurate = 100 * (total_acc / len(test_data))
        print('The total eval acccuracy : {:.3f}%'.format(total_accurate))