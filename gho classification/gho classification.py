dataset = 'kaggle.com/datasets/kumarajarshi/life-expectancy-who'

data_dir = './life-expectancy-who'

import os
os.listdir(data_dir)
pip install skillsnetwork[regular]
from tqdm import tqdm
pip install opendatasets --upgrade
import opendatasets as od
od.download(dataset)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def warn(*args,**kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

data = pd.read_csv('Life Expectancy Data.csv')
print(data.sample(5))

data.rename(columns={data.columns[3]:'Life expectancy',
                     data.columns[4]:'Adult mortality',
                     data.columns[5]:'Infant deaths',
                     data.columns[7]:'Health expenditure',
                     data.columns[9]:'Measles',
                     data.columns[10]:'BMI',
                     data.columns[11]:'Under-5 deaths',
                     data.columns[14]:'Diphtheria',
                     data.columns[15]:'HIV/AIDS',
                     data.columns[18]:'Thinness (1-19 years)',
                     data.columns[19]:'Thinness (5-9 years)'},
                    inplace=True)

data = data.drop('Country',axis=1)
first = data.pop('Status')
data.insert(len(data.columns),'Status',first)

null_count = data.isnull().sum()
print(null_count[null_count>0].sort_values(ascending=False),
      'Number of null entries:',null_count.sum())

for column in data:
    if data[column].isnull().sum() > 0:
        median = data[column].median()
        data[column].fillna(median,inplace=True)
null_count = data.isnull().sum()

print('Number of duplicated rows:',data.duplicated().sum())

print(data.sample(5))

x_cols = data.columns[:-1]
y_col = 'Status'
x_data = data[x_cols]
y_data = data[y_col]

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.25,random_state=42)

print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

poly = PolynomialFeatures(degree=2,include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
x_poly = poly.transform(x_data)

print(x_train_poly.shape[1],'features')

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_train_s = pd.DataFrame(x_train,columns=x_data.columns)
x_test = ss.transform(x_test)
x_test_s = pd.DataFrame(x_test,columns=x_data.columns)

print(x_train_s.sample(5))

# (UNREGULARISED) LOGISTIC REGRESSION
lr = LogisticRegression()
lr.fit(x_train_s,y_train)
y_pred_lr = lr.predict(x_test_s)
cm_lr = confusion_matrix(y_test,y_pred_lr)
plt.figure(figsize=(6,6))
sns.heatmap(cm_lr,annot=True,fmt='d',cmap='Reds',
            xticklabels=['Developed','Developing'], 
            yticklabels=['Developed','Developing'])
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.title('Logistic Regression Confusion Matrix')

plt.show()

# L1-REGULARISED LOGISTIC REGRESSION
lr_l1 = LogisticRegressionCV(Cs=10,cv=4,penalty='l1',solver='liblinear')
lr_l1.fit(x_train_s,y_train)
y_pred_lr_l1 = lr_l1.predict(x_test_s)
cm_lr = confusion_matrix(y_test,y_pred_lr_l1)
plt.figure(figsize=(6,6))
sns.heatmap(cm_lr,annot=True,fmt='d',cmap='Reds', 
            xticklabels=['Developed','Developing'], 
            yticklabels=['Developed','Developing'])
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.title('L1 Logistic Regression Confusion Matrix')

plt.show()

# L2-REGULARISED LOGISTIC REGRESSION
lr_l2 = LogisticRegressionCV(Cs=10,cv=4,penalty='l2',solver='liblinear')
lr_l2.fit(x_train_s,y_train)
y_pred_lr_l2 = lr_l2.predict(x_test_s)
cm_lr = confusion_matrix(y_test,y_pred_lr_l2)
plt.figure(figsize=(6,6))
sns.heatmap(cm_lr,annot=True,fmt='d',cmap='Reds', 
            xticklabels=['Developing','Developed'], 
            yticklabels=['Developing','Developed'])
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.title('L2 Logistic Regression Confusion Matrix')

plt.show()

coef = lr.coef_
coef_df = pd.DataFrame({'Feature':x_train_s.columns,'Coefficient':coef[0]}).sort_values(by='Coefficient',ascending=False)

display(coef_df)

# K-NEAREST NEIGHBOURS
neighbors = [1,2,3,4]
errors = []
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,12))
for i, neighbor in enumerate(neighbors):
    row = i // 2
    col = i % 2
    knc = KNeighborsClassifier(n_neighbors=neighbor)
    knc.fit(x_train,y_train)
    y_pred_knc = knc.predict(x_test)   
    cm_knc = confusion_matrix(y_test,y_pred_knc)
    sns.heatmap(cm_knc, annot=True,fmt='d',cmap='Blues',ax=ax[row,col], 
                xticklabels=['Developed','Developing'], 
                yticklabels=['Developed','Developing'])
    ax[row,col].set_title(f'{neighbor} Nearest Neighbours Confusion Matrix')
    ax[row,col].set_xlabel('Predictions')
    ax[row,col].set_ylabel('Ground Truth')

plt.show()

# RANDOM FOREST
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
plt.figure(figsize=(6,6))
sns.heatmap(cm_rfc, annot=True,fmt='d',cmap='Greens', 
            xticklabels=['Developed','Developing'], 
            yticklabels=['Developed','Developing'])
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.title('Random Forest Confusion Matrix')

plt.show()

# EXTRA TREES
etc = ExtraTreesClassifier()
etc.fit(x_train,y_train)
y_pred_etc = etc.predict(x_test)
cm_etc = confusion_matrix(y_test,y_pred_etc)
plt.figure(figsize=(6,6))
sns.heatmap(cm_etc,annot=True,fmt='d',cmap='Greens', 
            xticklabels=['Developed','Developing'], 
            yticklabels=['Developed','Developing'])
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.title('Extra Trees Confusion Matrix')

plt.show()

fi = rfc.feature_importances_
fi_df = pd.DataFrame({'Feature':x_train_s.columns,'Importance':fi}).sort_values(by='Importance',ascending=False)

display(fi_df)

target_names = ['developed','developing']
print(classification_report(y_test,y_pred_etc,target_names=target_names))