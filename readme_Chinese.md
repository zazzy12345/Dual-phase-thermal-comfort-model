## **说明：**

本文件夹包含二阶段热舒适度预测方法的相关代码和数据集，具体使用规则如下：

##### 	1. 第一阶段模型训练：

先利用phase1文件夹下的train.py训练第一阶段预测模型，使用的数据集为dataset/phase1Dataset下的图片数据集，本文件夹中已将其按照训练集：验证集：测试集以6：2：2的比例进行分类（后续可以将该数据集重新整合打乱）。

##### 	2. 组合数据集构建：

将得到的第一阶段预测模型，通过build_dataset.py文件进行组合数据集构建。

##### 	3. 第二阶段模型训练及预测：

将组合数据集输入phase2文件夹下的mlAlgorithm.py，选择合适的机器学习分类算法，完成第二阶段的模型构建和预测。

## phase1文件结构：

```
  ├── model.py: SAM-ResNet34模型搭建
  ├── train.py: 训练SAM-ResNet34的Phase1预测模型
  ├── predict.py: 单张图像预测脚本
  ├── build_dataset.py: 通过Phase1训练得到的预测模型构建Phase2所需的数据集
  ├── addtional_functions├── confusion_matrix_main.py: 混淆矩阵输出
 						 ├── data_augmentation.py: 图片数据集扩充
 						 └── read_txt.py: 辅助构建Phase2数据集相关函数
  └── model_result: 保存第一阶段的预测模型结果
```

## phase2文件结构：

```
  ├── mlAlgorithm.py: 通过机器学习算法完成最终的热舒适度预测结果
```

## dataset文件结构：

```
  ├── Phase1Dataset: Phase1阶段图片数据集
  					├── summer_thermal_image_224_random_622: 夏季6:2:2图片数据集
 				    └── winter_thermal_image_224_random_622: 冬季6:2:2图片数据集
  └── Phase2Dataset: Phase2阶段环境数据集+与第一阶段结合的组合数据集
  					├── ml_dataset: 组合数据集
    				├── summer_additional_feature_all: 夏季所有图片的环境数据（txt文件）
 				    └── winter_additional_feature_all: 冬季所有图片的环境数据（txt文件）
 				  	说明：summer_feature_fusion_622_test.csv等为组合数据集的范例
```

