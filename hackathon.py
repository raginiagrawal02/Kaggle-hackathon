import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score

df_train=pd.read_csv("train.csv")

le = preprocessing.LabelEncoder()
for col in df_train.columns:
    le.fit(df_train[col].values)
    df_train[col]=le.transform(df_train[col].values)
df_train=df_train.sample(frac=1).reset_index(drop=True)

feature_imp = pd.Series(clf.feature_importances_,index=df_train.drop(['Attrition','Id','PerformanceRating','Behaviour'],axis='columns').columns).sort_values(ascending=False)

X_train=df_train.drop(['Attrition','Id','PerformanceRating','Behaviour'],axis='columns')
y_train=df_train['Attrition']

df_test=pd.read_csv('test.csv')

for col in df_test.columns:
    le.fit(df_test[col].values)
    df_test[col]=le.transform(df_test[col].values)
X_test=df_test.drop(['Id','PerformanceRating','Behaviour'],axis='columns')


X_train,X_test,y_train,y_test=train_test_split(X_train, y_train, test_size=0.3)


clf=RandomForestClassifier(n_estimators=200,max_depth=9, n_jobs=-1,random_state=42)
clf.fit(X_train,y_train)

y_pred=clf.predict_proba(X_test)

print(roc_auc_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


