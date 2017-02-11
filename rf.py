# Random Forest Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

def test(model, X):
    Y = model.predict(X);
    print Y
    return Y

def evaluate(model, X, Y):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print results.mean()

def init():
    num_trees = 100
    max_features = "auto"
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    return model

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# trainRandomForest(X,Y)

