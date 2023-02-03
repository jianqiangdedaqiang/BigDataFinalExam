from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier

class MyModel:
    def __init__(self):
        self.bayes = None
        self.MLP = None
        self.SVM = None
        self.KNN = None
        self.SGD = None
        self.Ridge = None
        self.VoteClf = None

    def fit(self, x_train, y_train):
        self.bayes = GaussianNB()
        self.MLP = MLPClassifier(hidden_layer_sizes=(30, 20, 10), activation='relu', solver='sgd',
                                 learning_rate_init=0.001)
        self.SVM = svm.SVC()
        self.KNN = KNeighborsClassifier()
        # self.SGD = SGDClassifier()
        # self.Ridge = RidgeClassifier()
        self.VoteClf = VotingClassifier(estimators=[
            ("bys", self.bayes), ("MLP", self.MLP),
            ("svm", self.SVM), ("knn", self.KNN)], voting="hard")
        # self.VoteClf = VotingClassifier(estimators=[
        #     ("bys", self.bayes), ("MLP", self.MLP),
        #     ("svm", self.SVM), ("knn", self.KNN),
        #     ("SGD", self.SGD), ("Ridge", self.Ridge)], voting="hard")
        self.VoteClf.fit(x_train, y_train)
        print("训练完成")

    def predict(self, X_test):
        return self.VoteClf.predict(X_test)