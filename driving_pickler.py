from mysklearn.myclassifiers import MyNaiveBayesClassifier
import os
from mysklearn.mypytable import MyPyTable
import mysklearn.myevaluation as myevaluation
import mysklearn.myutils as myutils
import pickle

fname = os.path.join("input_data", "collisions.csv")
collisions_data = MyPyTable().load_from_file(fname)

weather = collisions_data.get_column('WEATHER')
road_condition = collisions_data.get_column('ROADCOND')
light_condition = collisions_data.get_column('LIGHTCOND')
junction_type = collisions_data.get_column('JUNCTIONTYPE')
severity = collisions_data.get_column('SEVERITYDESC')

X_train = [[weather[i],road_condition[i],light_condition[i],junction_type[i],severity[i]] for i in range(len(weather))]
y_train = collisions_data.get_column('COLLISIONTYPE')

for i,val in enumerate(y_train):
    if val == 'Unknown':
        del y_train[i]
        del X_train[i]

strattrain_folds, strattest_folds = myevaluation.stratified_kfold_cross_validation(X_train, y_train, 10)
strat_xtrain, strat_ytrain, strat_xtest, strat_ytest = myutils.get_from_folds(X_train, y_train, strattrain_folds, strattest_folds)

myb = MyNaiveBayesClassifier()
myb.fit(strat_xtrain, strat_ytrain)

packaged_object = myb
# pickle packaged object
outfile = open('driving_bayes.p', 'wb')
pickle.dump(packaged_object, outfile)
outfile.close()
