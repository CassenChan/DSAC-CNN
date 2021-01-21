import torch
from torchvision import transforms
from PIL import Image
from DSACNet import DSACNet

if __name__ == '__main__':

    classes = ['左偏67°', '左偏45°', '左偏22°', '正视00°', '右偏22°', '右偏45°', '右偏67°']
    weight_path = r'D:\PycharmProjects\PoseEstimate\Weights'
    img_path = r'C:\Users\w10\Desktop\1.jpg'
    pic_resize = (128, 128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = DSACNet().to(device)
    net.load_state_dict(torch.load(weight_path))  # 加载模型
    net.eval()

    transform = transforms.Compose([
        transforms.Resize(pic_resize), # 调整图片大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        images = img.to(device)
        output = net(images)

        _, predicted = torch.max(output.data, 1)
        class_index = predicted.cpu().numpy()[0]
        print('The class is {}'.format(classes[class_index]))




