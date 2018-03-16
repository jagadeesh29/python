import pandas as pd
import numpy as np
#import train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#copy train and test data
train_data = train
test_data = test

train_data.columns
test_data.columns
train_data.head(10)
train_data.tail(10)
#checking if there is any null value
null_check = train_data.isnull().sum()
train_summary = train_data.describe()

#removing varibales that has constant values
remove = []

for col in train_data.columns:
    if train_data[col].std() == 0:
        remove.append(col)

train_data.drop(remove, axis=1, inplace=True)
test_data.drop(remove, axis=1, inplace= True)


y_train = train_data['TARGET'].values
x_train = train_data.drop(['ID','TARGET'], axis =1).values

test_ID = test_data['ID'].values
x_test = test_data.drop(['ID'],axis =1).values

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=20, random_state= 123)
model_rf = random_forest.fit(x_train,y_train)
pred_rf =model_rf.predict_proba(x_test)[:,1]
pd.DataFrame({'ID':test_ID, 'TARGET':pred_rf}).to_csv('submission_rf.csv',index = False)

#randomforest accuracy= 0.710311
# =============================================================================
# from sklearn.svm import SVC
# svc =SVC(C=1.0,kernel = 'rbf')
# model_svm = svc.fit(x_train,y_train)
# pred_svm = model_svm.predict(x_test)
# pd.DataFrame({'ID':test_ID, 'TARGET':pred_svm}).to_csv('submission_svm.csv',index = False)
# 
# =============================================================================
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation = 'relu', solver='adam', random_state=123, learning_rate='adaptive')
model = mlp.fit(x_train,y_train)
y_pred_mlp = model.predict_proba(x_test)[:,1]
 
from sklearn.metrics import roc_auc_score
print('Overall AUC:', roc_auc_score(y_train, model.predict_proba(x_train)[:,1]))  

pd.DataFrame({'ID':test_ID, 'TARGET':y_pred_mlp}).to_csv('submission_NN.csv',index = False)

#by using MLPclassifier, kaggle accuracy= 0.809120
#before scaling it gves me 0.63 accuracy

