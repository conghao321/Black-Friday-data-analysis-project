import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
import logging
from sklearn.externals import joblib#保存模型 use this to save models
#logging :to record the results 
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("20181219_parameters.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Start print log")
#
#read the data
df=pd.read_csv('train.csv')
print(df.columns)

#data preprocessing

#map the categories
df['Age']=df['Age'].map({'0-17':0, '55+':6, '26-35':2, '46-50':4, '51-55':5, '36-45':3, '18-25':1})
df['Gender']=df['Gender'].map({'M':0,'F':1})
df['City_Category']=df['City_Category'].map({'A':0,'B':1,'C':2})
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].map({'2':2, '4+':4, '3':3, '1':1, '0':0})
df['Product_Category_2'].fillna(0,inplace=True)#填充缺失值
df['Product_Category_3'].fillna(0,inplace=True)   
le1=LabelEncoder()
df['Product_ID']=le1.fit_transform(df['Product_ID'])

#user ids are useless, so we drop them
#df.drop(['Occupation'],inplace=True,axis=1)
df.drop(['User_ID'],inplace=True,axis=1)

#train-test
train=df.loc[0:400000] #index base
test=df.loc[400001:]

#original result of targets:
result_0=test['Purchase']
#and we drop the purchase of test data
test.drop(['Purchase'],inplace=True,axis=1)
target=['Purchase']

#select the other data to be used as train set
predictor=[x for x in train.columns if x not in target]
train_y=train[target]
women_train=df[df['Gender']==1]
children_train=df[df['Age']==0]

#compare different models
'''
###11111111   LinearRegression
#score: 13
clf=LinearRegression(normalize=True)
clf.fit(train[predictor],train[target])
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
print(score.mean())

joblib.dump(clf, "LR_model.m")
clf2 = joblib.load("LR_model.m")
pre=clf2.predict(test)
'''

'''
####222222222222 DecisionTreeRegressor
#score: 63
clf=DecisionTreeRegressor(random_state=0,max_depth=6)
clf.fit(train[predictor],train[target])
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
print(score.mean())   
'''
'''


#######333333333333333Random Forest
#70
clf = RandomForestRegressor(n_estimators = 300, oob_score = True, n_jobs = -1,random_state =50,
                               max_features = 'auto', min_samples_leaf = 75)
clf.fit(train[predictor],np.ravel(train_y,order='C'))
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
kf_total
score=cross_validation.cross_val_score(clf, train[predictor], np.ravel(train_y,order='C'), cv=kf_total, n_jobs=1)
print(score.mean()) 

'''

#xgboost:
#75


xgboost_model =XGBRegressor(
    objective = 'reg:linear',
    booster= 'gbtree',   
    n_estimators = 300,
    min_child_weight = 20,
    learn_rate = 0.05,
    max_depth = 12,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.7,
    reg_alpha = 1,
    nthread= 4,
    eta=1
)



kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
score=cross_validation.cross_val_score(xgboost_model, train[predictor], train[target], cv=kf_total, n_jobs=1)
print(score.mean())

women_kf_total = cross_validation.KFold(len(women_train), n_folds=10,shuffle=True, random_state=4)
score_women=cross_validation.cross_val_score(xgboost_model, women_train[predictor], women_train[target], cv=women_kf_total, n_jobs=1)
print(score_women.mean())

children_kf_total = cross_validation.KFold(len(children_train), n_folds=10,shuffle=True, random_state=4)
score_children=cross_validation.cross_val_score(xgboost_model, children_train[predictor], children_train[target], cv=children_kf_total, n_jobs=1)
print(score.mean())


xgboost_model.fit(train[predictor],train[target])

#save the model
joblib.dump(xgboost_model, "xgb_model.m")
#load the model
clf2 = joblib.load("xgb_model.m")
result=clf2.predict(test)

'''
logger.info('common:n_estimators=300 learn_rate=0.05:scoremean:'+str(score.mean()))
logger.info('women:n_estimators=300 learn_rate=0.05:scoremean:'+str(score_women.mean()))
logger.info('children:n_estimators=300 learn_rate=0.05:scoremean:'+str(score_children.mean()))
logger.removeHandler(handler)
'''

#result_visualization:
test_new=pd.DataFrame.copy(test,deep=True)
result=result.tolist()
test_new['Purchase']=result
test_old=df.loc[400001:]
#根据年龄和性别分组运算
#data
means_new=test_new['Purchase'].groupby([test_new['Gender'], test_new['Age']]).mean()
means_old=test_old['Purchase'].groupby([test_old['Gender'], test_old['Age']]).mean()
means_new_men=means_new[0:7]
means_new_women=means_new[7:14]
means_old_men=means_old[0:7]
means_old_women=means_old[7:14]


#Line graph
plt.figure(figsize=(20,12))
plt.tick_params(labelsize=23)
#男女在不同年齡層購買力
labels=np.array(["0-17","18-25","26-35","36-45","46-50","51-55","55+"])
x = ["0-17","18-25","26-35","36-45","46-50","51-55","55+"]#点的横坐标
k1 = means_new_men
k2 = means_new_women
k3 = means_old_men
k4 = means_old_women
plt.plot(x,k1,'s-',color = 'b',label="Predicted Male")#s-:方形
plt.plot(x,k2,'s-',color = 'y',label="Predicted Female")#s-:方形
plt.plot(x,k3,'o-',color = 'g',label="Former Male")#o-:圆形
plt.plot(x,k4,'o-',color = 'r',label="Former Female")#o-:圆形
#font
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}
plt.xlabel("Age")#横坐标名字
plt.ylabel("Purchase")#纵坐标名字
plt.legend(loc = "best",prop=font1)#图例
plt.show()

