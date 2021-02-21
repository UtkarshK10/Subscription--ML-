import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

dataset=pd.read_csv('new_appdata.csv')

#Data Preprocessing
response=dataset['enrolled']
dataset=dataset.drop(columns='enrolled')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset,response,test_size=0.2,random_state=0)

train_identifier=X_train['user']
X_train=X_train.drop(columns='user')
test_identifier=X_test['user']
X_test=X_test.drop(columns='user')


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train2=pd.DataFrame(sc_X.fit_transform(X_train))
X_test2=pd.DataFrame(sc_X.fit_transform(X_test))
X_train2.columns=X_train.columns.values
X_test2.columns=X_test.columns.values
X_train2.index=X_train.index.values
X_test2.index=X_test.index.values
X_train=X_train2
X_test=X_test2

#Model Building
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,penalty='l2')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
cm=confusion_matrix(y_test, y_pred)
accuracy_score(y_test,y_pred)#.7679
precision_score(y_test,y_pred)#.7608
recall_score(y_test,y_pred)#0.7717
f1_score(y_test,y_pred)#0.7661


df_cm = pd.DataFrame(cm,index=(0,1),columns=(0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm,annot=True,fmt='g')
print('Test Data Acuuracy: %0.4f' % accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print('Logistic Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(),accuracies.std()*2))

#Formatting the final Results
final_results=pd.concat([y_test,test_identifier],axis=1).dropna()
final_results['predicted_results']=y_pred
final_results[['user','enrolled','predicted_results']].reset_index(drop=True)