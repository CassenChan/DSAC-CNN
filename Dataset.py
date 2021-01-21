from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from DSACNet import DSACNet

class Posedataset(Dataset):
    def __init__(self, anno_path, img_path, transform=None):
        ids = []
        f = open(anno_path, 'r')
        for line in f:
            line = line.strip('\n')
            line = line.rstrip()
            imgs, labels = line.split()[0], int(line.split()[1])
            ids.append((imgs, labels))
        self.ids = ids
        self.transform = transform
        self.img_path = img_path

    def __getitem__(self, index):
        img_name, label = self.ids[index]
        img = Image.open(self.img_path +'/'+ img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    transform = transforms.Compose([
                                    transforms.Resize((32, 32)),  # 裁减至（256,512）
                                    transforms.RandomRotation((30, 150)),  # 随机旋转30至150度
                                    transforms.RandomHorizontalFlip(0.6),  # 水平翻转
                                    transforms.RandomVerticalFlip(0.4),  # 垂直翻转
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])
    anno_path = r'C:\Users\Cassen\PycharmProjects\PoseEstimate\data_real\anno_train.txt'
    img_path = r'C:\Users\Cassen\PycharmProjects\PoseEstimate\data_real\img_together'
    train_data = Posedataset(anno_path=anno_path,  img_path=img_path, transform=transform)
    data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    for i, (img, cond) in enumerate(data_loader):
        print(i,cond)
        # pass