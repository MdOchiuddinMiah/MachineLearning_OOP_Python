from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


class DataRetrive:
    filepath = "E:\\Python\Datasets\\weather_training.csv"  # class variable
    featute_col = ['outlook', 'temperature', 'humidity', 'windy']
    class_col = ["play"]

    def __init__(self, localPath, featute_col, class_col):
        self.localPath = localPath
        self.featute_col = featute_col
        self.class_col = class_col

    @classmethod
    def getpath(cls):
        return cls.filepath

    @classmethod
    def getfeatutecol(cls):
        return cls.featute_col

    @classmethod
    def getclasscol(cls):
        return cls.class_col

    def getdata(self):
        return pd.read_csv(self.filepath)

    def datasplit(self, data):
        # Spliting dataset 67% train & 33% test
        x_train, x_test, y_train, y_test = train_test_split(data[self.featute_col], data[self.class_col],
                                                            train_size=0.75, random_state=0)
        return x_train, x_test, y_train, y_test

    def getdatatype(self, data, col=None):
        return col is None and data.dtypes or data[col].dtype


class DataPreprocessing:
    class_col = ["yes", "no"]

    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test

    @classmethod
    def getclassvalue(cls):
        return cls.class_col

    def labelencoding(self, col, dtype):
        le = LabelEncoder()
        le.fit(self.x_train[col].astype(dtype))
        x_train[col] = le.transform(self.x_train[col].astype(dtype))
        x_test[col] = le.transform(self.x_test[col].astype(dtype))
        return x_train, x_test


class ClassificationPerformation(DataPreprocessing):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.y_train = y_train
        self.y_test = y_test
        super().__init__(x_train, x_test)

    def fitclassifier(self):
        clf = tree.DecisionTreeClassifier()
        # clf = RandomForestClassifier()
        clf = clf.fit(self.x_train, self.y_train)
        return clf

    def showperformance(self, clf, label):
        print('Accuracy of the training set: {:.2f}'.format(clf.score(self.x_train, self.y_train) * 100) + ' %')
        print('Accuracy of the test set: {:.2f}'.format(clf.score(self.x_test, self.y_test) * 100) + ' %')
        predicted = clf.predict(self.x_test)
        confusion = confusion_matrix(self.y_test.to_numpy(), predicted, labels=label)
        print(confusion)
        print(classification_report(y_test, predicted))


# call class
dataRetrive = DataRetrive(DataRetrive.getpath(), DataRetrive.getfeatutecol(), DataRetrive.getclasscol())
filedata = dataRetrive.getdata()
x_train, x_test, y_train, y_test = dataRetrive.datasplit(filedata)

classificationPerformation = ClassificationPerformation(x_train, x_test, y_train, y_test)
x_train, X_test = classificationPerformation.labelencoding('outlook', str)
clf = classificationPerformation.fitclassifier()
classificationPerformation.showperformance(clf, classificationPerformation.getclassvalue())
