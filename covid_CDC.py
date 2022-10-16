#!/usr/bin/env python
# coding: utf-8

# In[1005]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[1006]:


os.chdir(r'E:\MachineLearning\daata\covidCDC')


# In[1007]:


os.listdir()


# In[1008]:


df=pd.read_csv('covid19_sample (1).csv')


# In[1009]:


df.dropna(inplace=True)


# In[1010]:


df.head()


# In[1011]:


df.shape


# In[1012]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[1013]:


df.shape


# #### df1=df.sample(frac=.02)# for the extraction of fraction of data 

# from sklearn.model_selection import train_test_split
# df,_=train_test_split(df,test_size=.98,random_state=0)

# In[1014]:


# ys=df[df['death_yn'].isin(['Yes'])]
# ns=df[df['death_yn'].isin(['No'])]


# In[1015]:


# 1.here we are considering variable Race and ethnicity (combined)


# In[1016]:


df['Race and ethnicity (combined)'].value_counts(dropna=False)


# In[1017]:


df.rename(columns={'Race and ethnicity (combined)':'Race_ethinicity'},inplace=True)


# In[1018]:


df['Race_ethinicity'].value_counts(dropna=False)


# In[1019]:


df1=pd.get_dummies(data=df,columns=['Race_ethinicity'],drop_first=True)


# In[1020]:


df1.columns=df1.columns.str.replace(" ","_")
df1.columns=df1.columns.str.replace("-","_")
df1.columns=df1.columns.str.replace(",","_")
df1.columns=df1.columns.str.replace("/","_")


# In[1021]:


df1.head()


# In[1022]:


df.dtypes


# In[1023]:


# 2.Now considering variable cdc_report_dt


# In[1024]:


# cdc_report_dt
# q1,q2,q3,q4: jan-mar-q1;apr-jun-q2 etc
# month: is_start_m,end_month,mid_month,salary_day(1-5); second_salary_day (15-20)
# day: weekend,weekday,public_holiday
# hour:early morning,evening...etc
# 


# In[1025]:


df1['cdc_report_dt'].dtypes


# In[1026]:


df1['cdc_report_dt']=pd.to_datetime(df1['cdc_report_dt'])


# In[1027]:


df1['cdc_report_dt'].dtypes


# In[1028]:


# here we find date format
pd.DataFrame(df1['cdc_report_dt']).iloc[1:3,:]


# In[1029]:


# code to extract quater from date ie cdc_report_dt
pd.DataFrame(df1.cdc_report_dt.dt.quarter).iloc[1:3,:]


# In[1030]:


# 2.1Finding quarter from cdc_report_dt


# In[1031]:


df1['Quarter']=df1.cdc_report_dt.dt.quarter # to create Quarter column from date column


# In[1032]:


df1['Quarter'].value_counts()


# In[1033]:


df1=pd.get_dummies(data=df1,columns=['Quarter'],drop_first=True)


# In[1034]:


df1.head(3)


# In[1035]:


#2.2 way to get monthly category
df1['mon_cat']=df1.cdc_report_dt.dt.day


# In[1036]:


df1['mon_cat'].value_counts()


# In[1037]:


# function to apply on mon_cat varible
def Mo_cat(i):
    if i in range(1,11):
        return 'Month_Start'
    elif i in [25,26,27,28,29,30,31]:
        return 'Month_End'
    else:
        return 'Month_Mid'


# In[1038]:


df1['mon_cat']=df1['mon_cat'].apply(Mo_cat)


# In[1039]:


df1['mon_cat'].value_counts()


# In[1040]:


df1=pd.get_dummies(data=df_final1,columns=['mon_cat'],drop_first=True)


# In[1041]:


df1.head()


# In[1102]:


# 2.3 to get whether day is weekday or weekend
df1['Day_Cat']=df1.cdc_report_dt.dt.weekday 


# In[1103]:


def day_cate(i):
    if i in [0,1,2,3,4]:
        return 'Weekday'
    else:
        return 'Weekend'


# In[1104]:


df1['Day_Cat'].value_counts()


# In[1105]:


df1['Day_Cat']=df1['Day_Cat'].apply(day_cate)


# In[1106]:


df1['Day_Cat'].value_counts()


# In[1107]:


df1=pd.get_dummies(data=df1,columns=['Day_Cat'],drop_first=True)


# In[1108]:


df1.head(4)


# In[1049]:


# 3 now variable is 'current_status'


# In[1050]:


df1['current_status'].value_counts()


# This has two categories therefore converting into binary 

# In[1051]:


df1=pd.get_dummies(data=df1,columns=['current_status'],drop_first=True)


# In[1052]:


df1.columns=df1.columns.str.replace("-","_")
df1.columns=df1.columns.str.replace(" ","_")


# In[1053]:


df1.head(3)


# In[1054]:


# 4 now variable is 'sex'


# In[1055]:


df1['sex'].value_counts()


# since there are in general three categories in sex variable therfore combining the unkown and Missing cat into one

# In[1056]:


df1['sex']=np.where(df1['sex']=='Male','Male',
                    np.where(df1['sex']=='Female','Female','others'))


# In[1057]:


df1['sex'].value_counts()


# In[1058]:


df1=pd.get_dummies(data=df1,columns=['sex'],drop_first=True)


# In[1059]:


df1.head()


# In[1060]:


# 5 here variable is age_group
df1['age_group'].value_counts()


# In[1061]:


df1['age_code']=pd.factorize(df1.age_group)[0]


# In[1062]:


df1.loc[:,['age_group','age_code']].value_counts()


# In[1063]:


def age_cat(x):
    if x in [1,0,3]:
        return 'SrCitz'
    elif x in [2,4]:
        return 'Adult'
    elif x in [6,5]:
        return 'Young'
    elif x in [7,8]:
        return 'Chdr&teen'
    elif x == 9:
        return 'Unkown'


# In[1064]:


df1['age_cate']=df1['age_code'].apply(age_cat)


# In[1065]:


df1=pd.get_dummies(data=df1,columns=['age_cate'],drop_first=True)


# In[1066]:


# 6 here we are considering variable name as hosp_yn


# In[1067]:


df1['hosp_yn'].value_counts()


# In[1068]:


df1['hospit']=np.where(df1['hosp_yn']=='No','NO',
                        np.where(df1['hosp_yn']=='Yes','YES','Unkown'))


# In[1069]:


df1['hospit'].value_counts()


# In[1070]:


df1=pd.get_dummies(data=df1,columns=['hospit'],drop_first=True)


# In[1071]:


# 7 now the variabel is icu_yn


# In[1072]:


df1['icu_yn'].value_counts()


# In[1073]:


df1['In_ICU']=np.where(df1['icu_yn']=='No','NO',
                        np.where(df1['icu_yn']=='Yes','YES','Unkown'))


# In[1074]:


df1['In_ICU'].value_counts()


# In[1075]:


df1=pd.get_dummies(data=df1,columns=['In_ICU'],drop_first=True)


# In[1076]:


# 8 now the variabel is death_yn


# In[1077]:


df1['death_yn'].value_counts()


# In[1078]:


df1=pd.get_dummies(data=df1,columns=['death_yn'],drop_first=True)


# In[1079]:


# 9 now the variabel is medcond_yn


# In[1080]:


df1['medcond_yn'].value_counts()


# In[1081]:


df1['MediCondi']=np.where(df1['medcond_yn']=='No','NO',
                        np.where(df1['medcond_yn']=='Yes','YES','Unkown'))


# In[1082]:


df1['MediCondi'].value_counts()


# In[1083]:


df1=pd.get_dummies(data=df1,columns=['MediCondi'],drop_first=True)


# In[1084]:


df1.head()


# In[1085]:


df1.dtypes


# In[1086]:


df1.columns


# In[1087]:


df2=df1[['Race_ethinicity_Asian__Non_Hispanic',
       'Race_ethinicity_Black__Non_Hispanic',
       'Race_ethinicity_Hispanic_Latino', 'Race_ethinicity_Missing',
       'Race_ethinicity_Multiple_Other__Non_Hispanic',
       'Race_ethinicity_Native_Hawaiian_Other_Pacific_Islander__Non_Hispanic',
       'Race_ethinicity_Unknown', 'Race_ethinicity_White__Non_Hispanic',
       'Quarter_2', 'Quarter_3', 'Quarter_4',
       'mon_cat_Month_Mid', 'mon_cat_Month_Start', 'Day_Cat_Weekend',
       'current_status_Probable_Case', 'sex_Male', 'sex_others',
       'age_cate_Chdr&teen', 'age_cate_SrCitz', 'age_cate_Unkown',
       'age_cate_Young', 'hospit_Unkown', 'hospit_YES', 'In_ICU_Unkown',
       'In_ICU_YES', 'death_yn_Yes', 'MediCondi_Unkown', 'MediCondi_YES'
       ]]


# In[1088]:


df2.head()


# In[1109]:


df2.dtypes


# ### Feature Engineering

# #### A. univariate Analysis

# In[1089]:


df.head()


# In[1090]:


df.dtypes


# In[984]:


def univariate_cat(data,x):
    missing=data[x].isnull().sum()
    unique_cnt=data[x].nunique()
    unique_cat=list(data[x].unique())
    
    f1=pd.DataFrame(data[x].value_counts(dropna=False))
    f1.rename(columns={'ed':'Count'},inplace=True)
    
    f2=pd.DataFrame(data[x].value_counts(normalize=True))
    f2.rename(columns={x:'Percentage'},inplace=True)
    f2['Percentage']=(f2['Percentage']*100).round(2).astype(str)+"%"
    ff=pd.concat([f1,f2],axis=1)    
    
    print(f"Total missing values : {missing}\n")
    print(f"Total count of unique categories: {unique_cnt}\n")
    print(f"Unique categories :\n{unique_cat}")
    print("value counts and %n ",ff)
    plt.figure(figsize=(5,4))
    sns.countplot(data=data,x=x)
    plt.show()


# In[985]:


univariate_cat(data=df,x='current_status')


# In[986]:


univariate_cat(data=df,x='sex')


# In[987]:


univariate_cat(data=df,x='age_group')


# In[988]:


univariate_cat(data=df,x='Race_ethinicity')


# In[989]:


univariate_cat(data=df,x='hosp_yn')


# In[990]:


univariate_cat(data=df,x='icu_yn')


# In[991]:


univariate_cat(data=df,x='death_yn')


# In[992]:


univariate_cat(data=df,x='medcond_yn')


# ### Bivariate Analysis

# In[993]:


df.columns


# #### Target variable in this situation is death_yn

# #### Target variable must have strong correlation between the feature but the feature themselves must not have strong correlation

# cat-cat<br>
# cat-num/num-cat<br>
# num-num<br>

# In[1093]:


# cat-cat
pd.crosstab(df['death_yn'],df['sex'])


# In[1095]:


pd.crosstab(df['death_yn'],df['Race_ethinicity'])


# In[1101]:


plt.figure(figsize=(10,6))
cr=df2.corr()
sns.heatmap(cr,annot=True,cmap='coolwarm')
plt.show()


# In[ ]:





# In[ ]:





# ### Model Development

# #### 1.Logistic Regression

# In[1100]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[1111]:


y=df2['death_yn_Yes']
x=df2.drop(columns=['death_yn_Yes'])


# In[1113]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,
                                              random_state=0)


# In[1114]:


logR=LogisticRegression()
logR.fit(x_train,y_train)


# In[1115]:


print('Train Score:',logR.score(x_train,y_train))
print('Test Score:',logR.score(x_test,y_test))


# In[1116]:


pred_train=logR.predict(x_train)
pred_train


# In[1117]:


act_pred_train=pd.DataFrame({'Act':y_train,'Pred':pred_train})
act_pred_train


# #### 1.1 Model Evaluation

# ##### confusion matrix

# In[1119]:


pd.crosstab(act_pred_train['Act'],act_pred_train['Pred'])


# In[1120]:


conf_train=metrics.confusion_matrix(y_train,pred_train)
conf_train


# In[1121]:


pd.DataFrame(conf_train,columns=["Pred_0_neg","Pred_1_pos"],
            index=['Act_0_neg','Act_1_pos'])


# In[1122]:


print(metrics.classification_report(y_train,pred_train))


# In[1123]:


prob_train=pd.DataFrame(logR.predict_proba(x_train),columns=['prob_0','prob_1'])


# In[1124]:


prob_train


# In[1125]:


new_pred_train=np.where(prob_train['prob_1']>.5,1,0)
print(metrics.classification_report(y_train,new_pred_train))


# ### 2.Decision Tree

# In[1126]:


from sklearn import metrics 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[1127]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[1128]:


print('Train Accuracy:',round(dt.score(x_train,y_train),2))
print('Test Accuracy:',round(dt.score(x_test,y_test),2))


# In[1129]:


from sklearn.tree import plot_tree
fn=x_train.columns
cn=['yes','no']
# Setting dpi= 300 to make image clearer than default
fig, axes=plt.subplots(nrows=1,ncols=1,figsize=(5,5), dpi=500)

dt_plot=plot_tree(dt,
                 feature_names=fn,
                 class_names=cn,
                 filled=True);


# In[1130]:


dt1=DecisionTreeClassifier(min_samples_split=150)
dt1.fit(x_train,y_train)
print('Train Accuracy:',str(round(dt1.score(x_train,y_train),3)*100)+str('%'))
print('Test Accuracy:',str(round(dt1.score(x_test,y_test),3)*100)+str('%'))


# In[1131]:


dt1=DecisionTreeClassifier(min_samples_leaf=50)
dt1.fit(x_train,y_train)
print('Train Accuracy:',str(round(dt1.score(x_train,y_train),3)*100)+str('%'))
print('Test Accuracy:',str(round(dt1.score(x_test,y_test),3)*100)+str('%'))


# In[1132]:


dt1=DecisionTreeClassifier(criterion='entropy')
dt1.fit(x_train,y_train)
print('Train Accuracy:',str(round(dt1.score(x_train,y_train),3)*100)+str('%'))
print('Test Accuracy:',str(round(dt1.score(x_test,y_test),3)*100)+str('%'))


# In[1134]:


dt1=DecisionTreeClassifier(criterion='gini')
dt1.fit(x_train,y_train)
print('Train Accuracy:',str(round(dt1.score(x_train,y_train),3)*100)+str('%'))
print('Test Accuracy:',str(round(dt1.score(x_test,y_test),3)*100)+str('%'))


# ### Grid Search and Random Search

# In[1135]:


from sklearn.model_selection import GridSearchCV

params={
    'criterion':['gini','entropy'],
    'max_depth':[5,7,9,10,11],
    'min_samples_split':[10,15,20,50,100,200,250],
    'min_samples_leaf':[5,10,15,20,50,80,100]}
dtg=DecisionTreeClassifier()
    
gd_search=GridSearchCV(estimator=dtg,param_grid=params,cv=10,n_jobs=-1,verbose=2)

gd_search.fit(x_train,y_train)
    


# In[1136]:


# k-fold cross validation:it is a technique of overfitting
# cv: Cross Validation =10
# for 10000 records it will divide into 10 parts since cv is 10
#     10000:1000*10


# In[1137]:


gd_search.best_params_


# In[1138]:


gd_search.best_score_


# In[1139]:


gd_search.best_estimator_


# In[1140]:


gd_search


# In[1141]:


pd.DataFrame(gd_search.cv_results_)


# In[1150]:


dt_f=DecisionTreeClassifier(criterion='gini',max_depth=11,
                           min_samples_leaf=10,
                           min_samples_split=100)
dt_f.fit(x_train,y_train)


# In[1151]:


from sklearn.tree import plot_tree
fn=x_train.columns
cn=['yes','no']
# Setting dpi= 300 to make image clearer than default
fig, axes=plt.subplots(nrows=1,ncols=1,figsize=(5,5), dpi=1000)

dt_plot=plot_tree(dt_f,
                 feature_names=fn,
                 class_names=cn,
                 filled=True);


# In[ ]:


#### Feature Importance


# In[1152]:


dt_f.feature_importances_


# In[1153]:


feat_imp=pd.DataFrame({'Variable':x_train.columns,
             'Imp':dt_f.feature_importances_}).sort_values(by='Imp',ascending=False)
feat_imp


# In[1155]:


feat_imp[feat_imp['Imp']>=0.01]


# In[1156]:


feat_imp[feat_imp['Imp']>=0.01]['Variable'].unique()


# In[1157]:


plt.figure(figsize=(15,5))
sns.barplot(data=feat_imp.head(10),x='Variable',y='Imp')


# In[1158]:


feat_imp[feat_imp['Imp']>=0.01]['Variable'].unique()


# In[1159]:


x_train1=x_train[['age_cate_SrCitz', 'hospit_YES', 'hospit_Unkown', 'Quarter_2']]
x_test1=x_test[['age_cate_SrCitz', 'hospit_YES', 'hospit_Unkown', 'Quarter_2']]


# In[1160]:


dt_f=DecisionTreeClassifier(criterion='gini',max_depth=11,
                           min_samples_leaf=10,
                           min_samples_split=100)
dt_f.fit(x_train1,y_train)
print('Train Accuracy:',str(round(dt_f.score(x_train1,y_train),3)*100)+str('%'))
print('Test Accuracy:',str(round(dt_f.score(x_test1,y_test),3)*100)+str('%'))


# In[1161]:


pred_train=dt_f.predict(x_train1)
pred_test=dt_f.predict(x_test1)


# In[1162]:


prob_train_1=dt_f.predict_proba(x_train1)[:,1]
prob_test_1=dt_f.predict_proba(x_test1)[:,1]


# In[1163]:


metrics.accuracy_score(y_train,pred_train)
metrics.accuracy_score(y_test,pred_test)


# In[1164]:


metrics.recall_score(y_train,pred_train)
metrics.recall_score(y_test,pred_test)


# In[1165]:


metrics.precision_score(y_train,pred_train)
metrics.precision_score(y_test,pred_test)


# In[1166]:


metrics.f1_score(y_train,pred_train)
metrics.f1_score(y_test,pred_test)


# In[1167]:


metrics.roc_auc_score(y_train,pred_train)


# In[1168]:


def classification_eva(act,pred,probs):
    ac1=metrics.accuracy_score(act,pred)
    rc1=metrics.recall_score(act,pred)    
    pc1=metrics.precision_score(act,pred)
    f1=metrics.f1_score(act,pred)
    auc1=metrics.roc_auc_score(act,pred)
    result={'Accuracy':ac1,'Recall':rc1,'Precision':pc1,'F1 score':f1,"AUC":auc1}
    
    fpr,tpr,threshold=metrics.roc_curve(act,probs)
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()
    return result


# In[1169]:


classification_eva(y_train,pred_train,probs=prob_train_1)


# In[1170]:


classification_eva(y_test,pred_test,probs=prob_test_1)


# In[ ]:





# #### Ensemble

# ### RandomForest

# In[1142]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV , train_test_split, RandomizedSearchCV


# In[1143]:


rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)


# In[1145]:


print('Train_score',rfr.score(x_train,y_train))
print('Test_score',rfr.score(x_test,y_test))


# In[ ]:





# In[1146]:


pred_train=rfr.predict(x_train)
pred_train


# In[1147]:


pred_test=rfr.predict(x_test)
pred_test


# In[1148]:


# for train data 
print("Train MSE", np.mean((pred_train-y_train)**(2)))
print("Train rmse", np.sqrt(np.mean((pred_train-y_train)**(2))))
print("Train MAE",  np.mean(np.abs(pred_train-y_train)))
print("Train MAPE",  np.mean(np.abs((pred_train-y_train)/y_train)))


# In[1149]:


# for test data 
print("Test MSE", np.mean((pred_test-y_test)**(2)))
print("Test rmse", np.sqrt(np.mean((pred_test-y_test)**(2))))
print("Test MAE",  np.mean(np.abs(pred_test-y_test)))
print("Test MAPE",  np.mean(np.abs((pred_test-y_test)/y_test)))


# In[ ]:




