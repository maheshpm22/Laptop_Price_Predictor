#!/usr/bin/env python
# coding: utf-8

# # Libraries import 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[112]:


df = pd.read_csv('laptop_data.csv')


# ## Sample data check 

# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.duplicated().sum()


# In[9]:


df.isnull().sum()


# # Unnamed column removed 

# In[10]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[11]:


df.head()


# ## GB and kg literal are removed 

# In[12]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[13]:


df.head()


# # object to float conversion
# 

# In[14]:


df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')


# In[15]:


df.info()


# In[16]:


import seaborn as sns


# In[17]:


sns.distplot(df['Price'])


# # Distribution due to company brand 

# In[18]:


df['Company'].value_counts().plot(kind='bar')


# In[19]:


sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# # Distribution due to type of laptop

# In[20]:


df['TypeName'].value_counts().plot(kind='bar')


# In[21]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[22]:


sns.distplot(df['Inches'])


# In[23]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# The size of the laptop dosen't give a much better idea about the price of the laptop, just a vague idea that low size laptops have lesser price and higher size laptops have large price 

# # Different types/ways in which 'Screen Resolution" is being displayed 

# In[24]:


df['ScreenResolution'].value_counts()


# # New column 'Touchscreen' formation 

# In[25]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[26]:


df.sample(5)


# In[27]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[28]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[29]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[30]:


df.head()


# In[31]:


df['Ips'].value_counts().plot(kind='bar')


# In[32]:


sns.barplot(x=df['Ips'],y=df['Price'])


# # x-axis and y-axis mein split 

# In[33]:


new = df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[34]:


df['X_res'] = new[0]
df['Y_res'] = new[1]


# In[35]:


df.sample(5)


# # WAY to get the x-resolution from the X_res column

# In[36]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[37]:


df.head()


# In[38]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


# In[39]:


df.info()


# In[40]:


df.corr()['Price']


# # New feature 'PPI' is introduced 

#       PPI ( Pixels per inch ) is the new feature which can be better measure to deal with the price of the laptop rather than the different x and y pixels and inches of the laptop. So, better feed the algo with the ppi feature. 

#    Formula used is   
#       
#    PPI = $ \frac{ \sqrt{ (X_{res})^2 + (Y_{res})^2)}}{Inches} $

# In[41]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[42]:


df.corr()['Price']


# In[43]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[44]:


df.head()


# In[45]:


df.drop(columns=['Inches','X_res','Y_res'],inplace=True)


# In[46]:


df.head()


# # How different kinds of CPU is written 

# In[47]:


df['Cpu'].value_counts()


# # Comma is removed and first three words are extracted

# In[48]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[49]:


df.head()


# In[50]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# # Function fetch processor is applied on CPU name column

# In[51]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[52]:


df.head()


# In[53]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[54]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[55]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[56]:


df.head()


# In[57]:


df['Ram'].value_counts().plot(kind='bar')


# In[58]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# # How various kinds of memory is written is displayed

# In[59]:


df['Memory'].value_counts()


# # Conversion of Memory to several columns 
# 
#     Here, Memory is converted to HDD, SDD, Flash storage and hybrid storage, so if some pc has mixed configuration like HDD + SSD then HDD and SSD column is filled and other two are kept 0. 

# In[60]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[61]:


df.sample(5)


# # Memory column is dropped 

# In[62]:


df.drop(columns=['Memory'],inplace=True)


# In[63]:


df.head()


# In[64]:


df.corr()['Price']


# In[65]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[66]:


df.head()


# # How GPU is written is different ways 

# In[67]:


df['Gpu'].value_counts()


# In[68]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[69]:


df.head()


# In[70]:


df['Gpu brand'].value_counts()


# In[71]:


df = df[df['Gpu brand'] != 'ARM']


# In[72]:


df['Gpu brand'].value_counts()


# In[73]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[74]:


df.drop(columns=['Gpu'],inplace=True)


# In[75]:


df.head()


# # How OPsys is written in different ways

# In[76]:


df['OpSys'].value_counts()


# In[77]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[78]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[79]:


df['os'] = df['OpSys'].apply(cat_os)


# In[80]:


df.head()


# In[81]:


df.drop(columns=['OpSys'],inplace=True)


# In[82]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[83]:


sns.distplot(df['Weight'])


#     Weight column is basically bimodal in nature and the correlation gives idea about its impact on the label price.

# In[84]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[85]:


df.corr()['Price']


# In[86]:


sns.heatmap(df.corr())


#     The more whittish are should be considered and on the columns should be removed to get better results.
#     
#     Here there are no whittish columns as such so no need to remove anything. 

# # Log Transform is applied to get Normal distribution

# In[87]:


sns.distplot(np.log(df['Price']))


# In[88]:


X = df.drop(columns=['Price'])
y = np.log(df['Price'])


# In[89]:


X


# # Log transformed values of price 
# 
#     - Keep in mind while deploying the model, the price needs to be done and exponential operation (IMP) otherwise it would give log value as the final predicted value.

# In[90]:


y


# # Train-test split 

# In[91]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[92]:


X_train


# In[108]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[109]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# ### Linear regression 
# 
#     This method is taught in 100-days of machine learning playlist, if any doubt refer that.
#     
#     Using of Pipeline module. 

# In[95]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Ridge Regression

# In[96]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Lasso Regression

# In[97]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### KNN

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Decision Tree

# In[99]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### SVM

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Random Forest
# 
#     There is more scope of varying the hyperparameters and tuning the model.

# In[103]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### ExtraTrees

# In[102]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### AdaBoost

# In[104]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Gradient Boost

# In[105]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### XgBoost

# In[287]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Voting Regressor

# In[106]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Stacking

# In[110]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Exporting the Model

# In[115]:


import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[114]:


df


# In[309]:


X_train


# In[116]:


python -m pip install ipykernel
python-m ipykernel install --user


# In[ ]:




