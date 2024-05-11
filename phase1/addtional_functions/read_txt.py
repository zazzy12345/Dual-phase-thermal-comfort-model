import torch
import os
import re


# @function:利用正则表达式删除字符串中的” (x)“部分
# @input:字符串
def remove_number_in_parentheses(string):
    modified_string = re.sub(r'\s\(\d+\)', '', string)

    return modified_string


# @function:利用图片地址获取 txt 地址
# @input:图片地址
def get_txt_path(img_file_path):
    # 替换\字符、文件名以及文件后缀
    new_string = img_file_path.replace("jpg", "txt")
    prefix = "extra-"

    split_parts = new_string.split(prefix, 1)
    result = prefix + split_parts[1]

    new_string = remove_number_in_parentheses(result)
    pre_folder = "../dataset/phase2Dataset/winter_additional_feature_all/"
    new_string = pre_folder + new_string

    return new_string


# @function:获取 txt 内数据，转化为一维数组
# @input:txt 地址
def read_additional_feature(file_path):
    # 检查传入路径是否存在
    assert os.path.exists(file_path), "read_additional_feature could not find the txt path {}".format(file_path)

    # 打开txt文件
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()  # 读取第一行数据

    # 将字符串按照空格分隔成列表，并转换为浮点数或整数类型
    data_list = [float(item) for item in first_line.split(',')]

    # 生成torch的一维向量
    tensor_data = torch.tensor(data_list)
    tensor_data = tensor_data.unsqueeze(0)  # 在维度0上插入一个维度

    return tensor_data


# @function:获取环境特征,拼接多个图片的辅助特征,用于tensor拼接
# @input:批量图片的地址集合
def get_additional_tensor_features(file_paths):
    additional_feature = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # 针对图片地址进行处理
    for path in file_paths:
        txt_path = get_txt_path(path)
        assert os.path.exists(txt_path), "{} extra function txt path does not exist.".format(txt_path)
        additional_feature = torch.concat((additional_feature, read_additional_feature(txt_path)), dim=0)
    additional_feature = additional_feature[1:]  # 剔除[0,0,...,0]

    return additional_feature


# @function:获取环境特征,拼接多个图片的辅助特征,用于普通数据
# @input:批量图片的地址集合
def get_additional_normal_features(file_paths):
    txt_path = get_txt_path(file_paths)
    with open(txt_path, 'r') as file:
        content = file.read()

    additional_feature = content.split(',')

    return additional_feature


# @function:获取环境特征,拼接单个图片的辅助特征
def get_single_additional_feature(file_paths):
    txt_path = get_txt_path(file_paths)
    assert os.path.exists(txt_path), "{} extra function txt path does not exist.".format(txt_path)
    additional_feature = read_additional_feature(txt_path)

    return additional_feature


def main():
    get_txt_path("")


if __name__ == '__main__':
    main()
