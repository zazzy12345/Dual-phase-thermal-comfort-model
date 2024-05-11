from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import csv


def thermal_comfort_classification(train_file, test_file):
    # 导入自定义数据集
    train_data = pd.read_csv(train_file, encoding="utf-8")
    train_data = train_data.dropna()
    test_data = pd.read_csv(test_file, encoding="utf-8")
    test_data = test_data.dropna()

    # 提取特征和标签
    x_train = train_data[["cool", "comfortable", "warm", "humidity", "head_area",
                          "top_area", "foot_area", "chest_area"]]
    y_train = train_data["TSV"]

    x_test = test_data[["cool", "comfortable", "warm", "humidity", "head_area",
                        "top_area", "foot_area", "chest_area"]]
    y_test = test_data["TSV"]

    # 创建分类器
    # KNeighborsClassifier()  # AdaBoostClassifier()  # RandomForestClassifier()  # svm.SVC()  # DecisionTreeClassifier
    clf = KNeighborsClassifier()

    # 训练模型
    clf.fit(x_train, y_train)

    # 预测测试集
    y_predict = clf.predict(x_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro')
    recall = recall_score(y_test, y_predict, average='macro')
    f_score = f1_score(y_test, y_predict, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1_score:", f_score)

    # 输出混淆矩阵
    cm = confusion_matrix(y_test, y_predict)
    cm = np.transpose(cm)
    print("Confusion Matrix:")
    print(cm)

    # 绘制混淆矩阵图像
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    plt.xlabel('真实类别', size=17, fontname='SimSun')
    plt.ylabel('预测类别', size=17, fontname='SimSun')
    plt.xticks(np.arange(len([-1, 0, 1])), [-1, 0, 1], size=17, fontname='Times New Roman')
    plt.yticks(np.arange(len([-1, 0, 1])), [-1, 0, 1], size=17, fontname='Times New Roman')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", size=17, fontname='Times New Roman')

    plt.tight_layout()
    plt.show()


def main():
    train_file = "../dataset/phase2Dataset/winter_feature_fusion_622_train.csv"
    test_file = "../dataset/phase2Dataset/winter_feature_fusion_622_test.csv"
    thermal_comfort_classification(train_file, test_file)


if __name__ == '__main__':
    main()

