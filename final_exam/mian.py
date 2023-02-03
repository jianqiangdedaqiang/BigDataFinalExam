from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from model import MyModel
import time


def data_generator(data_size):
    X, y = make_classification(n_samples=data_size, n_features=50,
                               n_classes=10, n_informative=40, random_state=20)
    X[:, 2] = X[:, 2] * 1000
    return X, y


#  预处理
def preprocess(data, y):
    #  minmax归一化
    # data = preprocessing.minmax_scale(data)
    #  高斯标准化
    sts = preprocessing.StandardScaler()
    data = sts.fit_transform(data)
    # lda = LDA(n_components=9)
    # data = lda.fit_transform(data, y)
    # pca = PCA(n_components=10)
    #  再使用PCA降维
    # PCA_data = pca.fit_transform(data)
    # return PCA_data
    return data


if __name__ == '__main__':
    dataSize = 50000
    X, y = data_generator(dataSize)
    start = time.time()
    X = preprocess(X, y)
    pre_time = time.time()
    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        model = MyModel()
        model.fit(X_train, y_train)
        ens_fit_time = time.time()
        y_pre = model.predict(X_test)
        end_ens_time = time.time()
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        sig_fit_time = time.time()
        y_pre2 = clf.predict(X_test)
        sig_end_time = time.time()
        acc = accuracy_score(y_test, y_pre)
        acc2 = accuracy_score(y_test, y_pre2)
        print(f"数据预处理时间:   {pre_time - start} s")
        print(f"集成模型训练时间: {ens_fit_time - pre_time} s")
        print(f"集成模型预测时间: {end_ens_time - ens_fit_time} s")
        print()
        print(f"单一模型训练时间: {sig_fit_time - end_ens_time} s")
        print(f"单一模型预测时间: {sig_end_time - sig_fit_time} s")
        print(f"集成分类器正确率:        {acc}")
        print(f"单一分类器(决策树)正确率: {acc2}")

