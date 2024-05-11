import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm  # 进度条可视化库


from model import resnet34
import torchvision.models.resnet
import datetime


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # RandomResizeCrop(224),随意裁剪成224×224
                                     transforms.RandomHorizontalFlip(),  # RandomHorizontalFlip(),依据概率p对图片进行水平翻转,p默认0.5
                                     transforms.ToTensor(),
                                     # ToTensor()将图片转化为向量的格式，由H×W×C的[0,255],转化为C×H×W的[0,1]，也由于归一化无法处理PIL图像，所以要用ToTensor
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        # Normalize(std=[c1,c2,c3],mean=[c1,c2,c3]),Normalize就是图片归一化,将输入归一化到[0.1],再用(x-mean)/std，分布到[-1,1]
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]由imagenet训练集中抽样计算出的，最优数值
        "val": transforms.Compose([transforms.Resize(224),  # 原——256
                                   transforms.CenterCrop(224),  # 原——224
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # 数据集要放在代码相同的文件路径下
    image_path = os.path.join(data_root, "CodeAndDataset", "dataset", "phase1Dataset", "summer_thermal_image_224_random_622")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "extra_train"),  # 训练集调用
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    {'cold':0, 'comfort':1, 'warm':2}
    # flower_list = train_dataset.class_to_idx
    sense_list = train_dataset.class_to_idx  # 将类别名转化为数字序号
    cla_dict = dict((val, key) for key, val in sense_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)  # indent=nums,nums表示再json文件中类别前的空格数
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "extra_val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # net = resnet50()
    # 迁移学习
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"  # 预训练模型地址
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    net.fc = nn.Linear(in_channel, 3)  # 改变类别数量
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100
    best_acc = 0.0
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    save_path = f"model_result/model_pth/resNet34_attention_{current_date}.pth"
    train_steps = len(train_loader)

    # 记录训练loss和验证acc
    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data

            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        train_loss.append(running_loss/train_steps)
        val_acc.append(val_accurate)
        with open(f"model_result/loss_accuracy_text/train_loss_attention_{current_date}.txt", 'w') as train_los:
            train_los.write(str(train_loss))

        with open(f"model_result/loss_accuracy_text/val_acc_attention_{current_date}.txt", 'w') as val_ac:
            val_ac.write(str(val_acc))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Best_acc = '+str(best_acc))
    print('Finished Training')


if __name__ == '__main__':
    main()
