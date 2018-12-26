import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

df=pd.read_csv('train.csv')
print(df.columns)

df['Age']=df['Age'].map({'0-17':0, '55+':6, '26-35':2, '46-50':4, '51-55':5, '36-45':3, '18-25':1})
df['Gender']=df['Gender'].map({'M':0,'F':1})
df['City_Category']=df['City_Category'].map({'A':0,'B':1,'C':2})
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].map({'2':2, '4+':4, '3':3, '1':1, '0':0})
sum(df['Product_Category_2'].isnull())
df['Product_Category_2'].fillna(0,inplace=True)#填充缺失值
df['Product_Category_3'].fillna(0,inplace=True)   

#df.drop(['Occupation'],inplace=True,axis=1)
df.drop(['User_ID'],inplace=True,axis=1)

#Describe()
print(df.columns)
purchase_overview=df['Purchase'].describe()
print(purchase_overview)

#font
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

#purchase_overview
plt.figure(figsize=(20,12))
plt.tick_params(labelsize=23)
sns.distplot(df['Purchase'])

#heatmap
corrmat = df.corr()
plt.figure(figsize=(20,12))
plt.tick_params(labelsize=23)
legend = plt.legend(prop=font1)
heat_map=sns.heatmap(corrmat, vmax=.4,square=True)
heat_map.figure.axes[-1].yaxis.label.set_size(30)

#visualization:
#befor this we have already extract the related categories' data as indepentd csv file,
#including fm.csv(for gender-age groups' mean values of purchase)
#and cities as well as occupation

#1-Gender and Age
plt.figure(figsize=(20,12))
plt.tick_params(labelsize=23)
#age groups
labels=np.array(["0-17","18-25","26-35","36-45","46-50","51-55","55+"])
#read the data 
df=pd.read_csv('fm.csv')
x = ["0-17","18-25","26-35","36-45","46-50","51-55","55+"]#点的横坐标
k1 = df.loc[0,labels].values#线1的纵坐标
k2 = df.loc[1,labels].values#线2的纵坐标
plt.plot(x,k1,'s-',color = 'r',label="Female")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="Male")#o-:圆形
plt.xlabel("Age")#横坐标名字
plt.ylabel("Purchase")#纵坐标名字
plt.legend(loc = "best",prop=font1)#图例
plt.show()

#radar chart:
#we use a radar chart to illustrate the relationship of occupation and cities with purchase.

#Occupation
#read the data
plt.style.use('ggplot')
df=pd.read_csv('occupation.csv')
#labels:
labels=np.array(["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"])
values=df.loc[1,labels].values
values2=df.loc[0,labels].values
#verges
N = len(values)
#set the angel of radar chart
angles=np.linspace(0, 2*np.pi, N, endpoint=False)
values=np.concatenate((values,[values[0]]))
values2=np.concatenate((values2,[values2[0]]))
angles=np.concatenate((angles,[angles[0]]))
fig=plt.figure(figsize=(20,12))
plt.tick_params(labelsize=40)
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2, label = 'Occupation Mediam')
ax.fill(angles, values, alpha=0.25)
ax.plot(angles, values2, 'o-', linewidth=2, label = 'Occupation Average')
ax.fill(angles, values2, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_ylim(0,11000)
plt.title('Occupation\n')
ax.grid(True)
plt.legend(loc = 'best',prop=font1)
plt.show()

#Cities
plt.style.use('ggplot')
df=pd.read_csv('city.csv')
labels=np.array(["A","B","C"])
values=df.loc[1,labels].values
values2=df.loc[0,labels].values
N = len(values)
angles=np.linspace(0, 2*np.pi, N, endpoint=False)
values=np.concatenate((values,[values[0]]))
values2=np.concatenate((values2,[values2[0]]))
angles=np.concatenate((angles,[angles[0]]))
fig=plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2, label = 'City Mediam')
ax.fill(angles, values, alpha=0.25)
ax.plot(angles, values2, 'o-', linewidth=2, label = 'City Average')
ax.fill(angles, values2, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_ylim(0,11000)
plt.title('City\n')
ax.grid(True)
plt.legend(loc = 'best',prop=font1)
plt.show()
