import csv
import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34
from addtional_functions.read_txt import get_single_additional_feature, get_additional_normal_features


def main():  # 构建模型预测类别百分比+环境参数的数据集
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    images_root = "../dataset/phase1Dataset/winter_thermal_image_224_random_622/extra_val/warm"
    assert os.path.exists(images_root), f"file: '{images_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(images_root, i) for i in os.listdir(images_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=3).to(device)

    # load model weights
    weights_path = "./model_result/model_pth/winter_resNet34_attention_epoch100_622.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 1  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            cool = predict[0][0]/1   # cool的预测百分比
            comfort = predict[0][1]/1  # comfortable的预测百分比
            warm = predict[0][2]/1  # warm的预测百分比
            cool = cool.item()
            comfort = comfort.item()
            warm = warm.item()

            # 获取对应环境数据
            additional_features = get_single_additional_feature(img_path)
            with open("../dataset/phase2Dataset/ml_dataset/winter_feature_fusion_622_train.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([1, round(cool, 4), round(comfort, 4), round(warm, 4),
                                 round(additional_features[0][0].item(), 2), round(additional_features[0][1].item(), 2),
                                 round(additional_features[0][2].item(), 2), round(additional_features[0][3].item(), 2),
                                 round(additional_features[0][4].item(), 2), round(additional_features[0][5].item(), 2),
                                 round(additional_features[0][6].item(), 2), round(additional_features[0][7].item(), 2),
                                 round(additional_features[0][8].item(), 2), round(additional_features[0][9].item(), 2),
                                 round(additional_features[0][10].item(), 2), round(additional_features[0][11].item(), 2),
                                 round(additional_features[0][12].item(), 2), round(additional_features[0][13].item(), 2),
                                 round(additional_features[0][14].item(), 2), round(additional_features[0][15].item(), 2),
                                 round(additional_features[0][16].item(), 2), round(additional_features[0][17].item(), 2),
                                 round(additional_features[0][18].item(), 2)])


if __name__ == '__main__':
    main()
