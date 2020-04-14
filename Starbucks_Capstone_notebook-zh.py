#!/usr/bin/env python
# coding: utf-8

# # 星巴克毕业项目
# 
# ### 简介
# 
# 这个数据集是一些模拟 Starbucks rewards 移动 app 上用户行为的数据。每隔几天，星巴克会向 app 的用户发送一些推送。这个推送可能仅仅是一条饮品的广告或者是折扣券或 BOGO（买一送一）。一些顾客可能一连几周都收不到任何推送。 
# 
# 顾客收到的推送可能是不同的，这就是这个数据集的挑战所在。
# 
# 你的任务是将交易数据、人口统计数据和推送数据结合起来判断哪一类人群会受到某种推送的影响。这个数据集是从星巴克 app 的真实数据简化而来。因为下面的这个模拟器仅产生了一种饮品， 实际上星巴克的饮品有几十种。
# 
# 每种推送都有有效期。例如，买一送一（BOGO）优惠券推送的有效期可能只有 5 天。你会发现数据集中即使是一些消息型的推送都有有效期，哪怕这些推送仅仅是饮品的广告，例如，如果一条消息型推送的有效期是 7 天，你可以认为是该顾客在这 7 天都可能受到这条推送的影响。
# 
# 数据集中还包含 app 上支付的交易信息，交易信息包括购买时间和购买支付的金额。交易信息还包括该顾客收到的推送种类和数量以及看了该推送的时间。顾客做出了购买行为也会产生一条记录。 
# 
# 同样需要记住有可能顾客购买了商品，但没有收到或者没有看推送。
# 
# ### 示例
# 
# 举个例子，一个顾客在周一收到了满 10 美元减 2 美元的优惠券推送。这个推送的有效期从收到日算起一共 10 天。如果该顾客在有效日期内的消费累计达到了 10 美元，该顾客就满足了该推送的要求。
# 
# 然而，这个数据集里有一些地方需要注意。即，这个推送是自动生效的；也就是说，顾客收到推送后，哪怕没有看到，满足了条件，推送的优惠依然能够生效。比如，一个顾客收到了"满10美元减2美元优惠券"的推送，但是该用户在 10 天有效期内从来没有打开看到过它。该顾客在 10 天内累计消费了 15 美元。数据集也会记录他满足了推送的要求，然而，这个顾客并没被受到这个推送的影响，因为他并不知道它的存在。
# 
# ### 清洗
# 
# 清洗数据非常重要也非常需要技巧。
# 
# 你也要考虑到某类人群即使没有收到推送，也会购买的情况。从商业角度出发，如果顾客无论是否收到推送都打算花 10 美元，你并不希望给他发送满 10 美元减 2 美元的优惠券推送。所以你可能需要分析某类人群在没有任何推送的情况下会购买什么。
# 
# ### 最后一项建议
# 
# 因为这是一个毕业项目，你可以使用任何你认为合适的方法来分析数据。例如，你可以搭建一个机器学习模型来根据人口统计数据和推送的种类来预测某人会花费多少钱。或者，你也可以搭建一个模型来预测该顾客是否会对推送做出反应。或者，你也可以完全不用搭建机器学习模型。你可以开发一套启发式算法来决定你会给每个顾客发出什么样的消息（比如75% 的35 岁女性用户会对推送 A 做出反应，对推送 B 则只有 40% 会做出反应，那么应该向她们发送推送 A）。
# 
# 
# # 数据集
# 
# 一共有三个数据文件：
# 
# * portfolio.json – 包括推送的 id 和每个推送的元数据（持续时间、种类等等）
# * profile.json – 每个顾客的人口统计数据
# * transcript.json – 交易、收到的推送、查看的推送和完成的推送的记录
# 
# 以下是文件中每个变量的类型和解释 ：
# 
# **portfolio.json**
# * id (string) – 推送的id
# * offer_type (string) – 推送的种类，例如 BOGO、打折（discount）、信息（informational）
# * difficulty (int) – 满足推送的要求所需的最少花费
# * reward (int) – 满足推送的要求后给与的优惠
# * duration (int) – 推送持续的时间，单位是天
# * channels (字符串列表)
# 
# **profile.json**
# * age (int) – 顾客的年龄 
# * became_member_on (int) – 该顾客第一次注册app的时间
# * gender (str) – 顾客的性别（注意除了表示男性的 M 和表示女性的 F 之外，还有表示其他的 O）
# * id (str) – 顾客id
# * income (float) – 顾客的收入
# 
# **transcript.json**
# * event (str) – 记录的描述（比如交易记录、推送已收到、推送已阅）
# * person (str) – 顾客id
# * time (int) – 单位是小时，测试开始时计时。该数据从时间点 t=0 开始
# * value - (dict of strings) – 推送的id 或者交易的数额
# 
# **注意：**如果你正在使用 Workspace，在读取文件前，你需要打开终端/命令行，运行命令 `conda update pandas` 。因为 Workspace 中的 pandas 版本不能正确读入 transcript.json 文件的内容，所以需要更新到 pandas 的最新版本。你可以单击 notebook 左上角橘黄色的 jupyter 图标来打开终端/命令行。  
# 
# 下面两张图展示了如何打开终端/命令行以及如何安装更新。首先打开终端/命令行：
# <img src="pic1.png"/>
# 
# 然后运行上面的命令：
# <img src="pic2.png"/>
# 
# 最后回到这个 notebook（还是点击橘黄色的 jupyter 图标），再次运行下面的单元格就不会报错了。

# In[23]:


import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# In[ ]:





# In[5]:


profile.head()


# In[6]:


plt.hist(profile.age); profile.describe()


# In[7]:


g1= sns.FacetGrid(profile, col="gender")
g1.map(plt.hist, "age")


# In[8]:


plt.hist(profile.income[profile.income.notna()]);


# In[9]:


g1= sns.FacetGrid(profile, col="gender")
g1.map(plt.hist, "income")


# In[10]:


profile.apply(axis=0,func=lambda x: x.isna().sum())


# In[11]:


profile[profile.age != 118].isna().sum()


# In[12]:


missing_people = profile.loc[profile.age == 118,:].id


# In[13]:


portfolio.head()


# In[14]:


portfolio.offer_type.unique()


# In[15]:


portfolio.difficulty.unique()


# In[16]:


portfolio.apply(axis=0,func=lambda x: x.isna().sum())


# In[17]:


transcript.head()


# In[18]:


np.sum(pd.Series(transcript.person.unique()).isin(pd.Series((profile.id.unique()))))


# In[19]:


transcript.event.unique()


# In[20]:


transcript.groupby('event').count().loc[:,:'person']


# In[21]:


transcript.apply(axis=0,func=lambda x: x.isna().sum())


# In[2]:


def clean_trascript(transcript_df): 
    """ 
    #    Clean up the transcript data frame and return it. 
    #    INPUT:
    #       transcript_df

    #    
    #    OUTPUT:
    #        offer_dataset:events about the offers
    #        spending_dataset: dataset for the events about the transactions
    """

    
    # Extracting the offers_id and amounts from the value columns of the data 
    transcript_clean = pd.DataFrame(transcript_df.value.tolist())
    transcript_clean.columns = transcript_clean.columns.str.replace("offer id","offer_id")
    s = transcript_clean.stack()
    transcript_clean = s.unstack()
    
    #Dummy the offer event and merge it to the dataset 
    transcript_clean = pd.get_dummies(data=transcript_clean.join(transcript_df),columns=['event'])
    
    # Drop the value column ,delete any duplicated records
    transcript_clean.drop('value', axis=1, inplace=True)
    transcript_clean.drop_duplicates(inplace=True)
    
    # Segregate the offers and transactions into  different datasets. 
    offer_dataset = transcript_clean[transcript_clean.event_transaction != 1].copy()
    
    spending_dataset = transcript_clean[transcript_clean.event_transaction == 1].copy()
    
    # Drop the unecessary columns from the offer_dataset 
    offer_dataset.drop(['amount','event_transaction','reward'], axis=1, inplace=True)
    
    # Select the columns related to the spending behavior into a seperate dataset. 
    spending_dataset = spending_dataset.filter(['person', 'time', 'amount']).copy()


    return offer_dataset, spending_dataset.reset_index(drop=True)


# In[3]:


def clean_portfolio(portfolio_df): 
    
    # Extract and dummy the channel useed for each offer 
    portfolio_clean = pd.concat([ portfolio_df,pd.get_dummies(portfolio_df.channels.apply(pd.Series).stack()).sum(level=0)], axis=1, sort=False)
    
    # Drop channels and reward columns now 
    portfolio_clean.drop(['channels', 'reward'], axis=1, inplace=True)
    
    return portfolio_clean


# In[4]:


def clean_profile(profile_df): 

    """ 
    Clean up the profile data frame and return it. This function also deals with missing data as discussed. 
    it also parse the date for membership into three columns , year, month , day 
    
    INPUT:
       profile_df: Information related to users.

    #    
    OUTPUT:
        profile_df: Cleaned information. 
    """
    # Deleting missing data [justification of decision is below]   
    profile_df = profile_df.loc[profile_df.age != 118]
    
    # Parsing the membership date into different Columns
    membership_column = profile_df.loc[:,('became_member_on')].astype(str)
    profile_df.loc[:,'member_year'] = membership_column.str[0:4].tolist()
    profile_df.loc[:,'member_month'] = membership_column.str[4:6].tolist()
    profile_df.loc[:,'member_day'] = membership_column.str[6:8].tolist()
    
    
     #Dummy the gender and merge it back to the dataset 
    profile_df = pd.concat([profile_df, pd.get_dummies(profile_df.gender)],axis=1, sort=False)
    
    
    # Changing the id to person to make the dataset consistent with others. 
    profile_df.rename(columns={'id':'person'}, inplace=True)
    profile_df.drop(columns=['became_member_on', 'gender'], inplace=True)
    
    return profile_df


# In[5]:


offer_dataset,_ = clean_trascript(transcript)


# In[26]:


offer_dataset.head()


# In[6]:


def gen_analytic_df(cln_profile, cln_offer, cln_portfolio):
    """ 
    Merge users profiles with offered they encountered and offer information into one 
    analytical Data Frame 
    
    INPUT:
       cln_profile: cleaned up profile data frame. 
       cln_offer: cleaned up offer information (exclude transactions)
       cln_portfolio: cleaned up portfolio data frame. 

    #    
    OUTPUT:
        final_df: a processed dataset for all the events related to the offers
        spending_dataset: a processed dataset for all the events related to the transactions
    """
    
    cln_offer = cln_offer.groupby(["person","offer_id",]).sum().reset_index()
    offer_port = cln_offer.merge(cln_portfolio,left_on = 'offer_id', right_on = 'id').drop('id', axis = 1)
    offer_port.rename(columns={"event_offer received":"offer_recieved", "event_offer viewed":"offer_viewed",
                           "event_offer completed":"offer_completed"}, inplace = True)
    
    final_df  = offer_port.merge(cln_profile, on = "person").reset_index()
    
  
    
    
    return final_df


# In[28]:


profile.head()


# In[7]:


clean_portfolio(portfolio)


# In[8]:


clean_profile(profile)


# In[9]:


offer_dataset,_ = clean_trascript(transcript)


# In[10]:


final_df = gen_analytic_df(clean_profile(profile), offer_dataset, clean_portfolio(portfolio))


# In[40]:


final_df.head()


# In[41]:


final_df.columns.values.tolist()


# In[11]:


def analyze_offer_success(df):
    
    """ 
    Analyze offer success based on the criteria provided in the note book. 
    
    INPUT:
       df: the analytical Data_frame we generated that has all the mereged datasets


    #    
    OUTPUT:
        final_df: Data_frame that has the success variable and cleaner DataFrame (Dropped unnecessary columns)
    """
    
    
    
    df.reset_index(drop=True)
    successful = []

    for i,item in df.iterrows():

        if(item['offer_type'] == 'informational'): 

            if(item["offer_recieved"] > 0) & (item["offer_viewed"] > 0): 
                successful.append(1)
            else: 
                successful.append(0)

        else:

            if (item["offer_recieved"] > 0) & (item["offer_viewed"] > 0) & (item['offer_completed'] > 0): 
                successful.append(1)

            else: 
                successful.append(0)
                
    final_df = pd.concat([df,pd.DataFrame(successful, columns=["success"])], axis=1)
    
    final_df.drop(columns=['offer_completed', 'offer_recieved', 'offer_viewed', 'time'], inplace=True)
    
    return final_df


# In[12]:


analytic_df = analyze_offer_success(final_df)


# In[37]:


analytic_df.head()


# In[38]:


def get_graph_gender(df):
    """ 
    calculate the rate of success for each gender in a specified dataset. 
    
    INPUT:
       df:analytical Data_frame we generated that has all the mereged datasets


    #    
    OUTPUT:
        graph: list with values for the bargraph 
    """
    
        
    
    # Normalizaing factor 
    numb_F = df.loc[df.F == 1, 'success'].count()
    numb_M = df.loc[df.M == 1, 'success'].count()
    numb_O = df.loc[df.O == 1, 'success'].count()
    
    #Calculating success
    success_F = df.loc[df.F == 1, 'success'].sum()
    success_M = df.loc[df.M == 1, 'success'].sum()
    success_O = df.loc[df.O == 1, 'success'].sum()
    graph_data = [success_F/numb_F, success_M/numb_M, success_O/numb_O]
    return graph_data


# In[39]:


graph_data = get_graph_gender(analytic_df)


# In[40]:


# Create bars
plt.bar(x=np.arange(3),height = graph_data)
# Create names on the x-axis
plt.xticks(np.arange(3), ['F', 'M', 'O']);
plt.title("All Offers Success across genders")


# In[41]:


fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(8, 8))

ax1.bar(x=np.arange(3),height = get_graph_gender(analytic_df.loc[analytic_df.offer_type == "informational",:]))
ax1.set_title("informational Offers Success across genders")
ax2.bar(x=np.arange(3),height = get_graph_gender(analytic_df.loc[analytic_df.offer_type == "discount",:]))
ax2.set_title("discount Offers Success across genders")
ax3.bar(x=np.arange(3),height = get_graph_gender(analytic_df.loc[analytic_df.offer_type == "bogo",:]))
ax3.set_title("bogo Offers Success across genders")
ax1.set_xticks(np.arange(3)); ax1.set_xticklabels(['F', 'M', 'O'])
ax2.set_xticks(np.arange(3)); ax2.set_xticklabels(['F', 'M', 'O'])
ax3.set_xticks(np.arange(3)); ax3.set_xticklabels(['F', 'M', 'O'])
plt.subplots_adjust( hspace=.5)
plt.show()


# In[42]:


fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(8, 8))

ax1.bar(x=np.arange(3),height = get_graph_gender(analytic_df.loc[(analytic_df.age <= 35) ,:]))
ax1.set_title("Offers Success for young users")
ax2.bar(x=np.arange(3),height = get_graph_gender(analytic_df.loc[(analytic_df["age"] > 35) & (analytic_df["age"] <= 55),:]))
ax2.set_title("Offers Success for middle-age users")
ax3.bar(x=np.arange(3),height = get_graph_gender(analytic_df.loc[(analytic_df.age > 55),:]))
ax3.set_title("Offers Success for older users")
ax1.set_xticks(np.arange(3)); ax1.set_xticklabels(['F', 'M', 'O'])
ax2.set_xticks(np.arange(3)); ax2.set_xticklabels(['F', 'M', 'O'])
ax3.set_xticks(np.arange(3)); ax3.set_xticklabels(['F', 'M', 'O'])
plt.subplots_adjust( hspace=.5)
plt.show()


# In[13]:


model_bagging = BaggingClassifier(n_estimators = 250)

model_randomForest = RandomForestClassifier(n_estimators = 250)

model_ada = AdaBoostClassifier(n_estimators = 300, learning_rate=0.2)


# In[14]:


df = pd.concat([analytic_df,pd.get_dummies(analytic_df.offer_type)],axis = 1 ,sort=False)
df.reset_index(drop= True, inplace = True)


# In[49]:


df.head()


# In[64]:


x_features = ['difficulty','duration','email','mobile','social','F', 'M','O', 'bogo', 'discount', 'informational']


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(df[x_features], 
                                                    df['success'], 
                                                    random_state=1)


# In[66]:


model_bagging.fit(X_train,y_train)
model_randomForest.fit(X_train,y_train)
model_ada.fit(X_train,y_train)


# In[67]:


bag_pred = model_bagging.predict(X_test)
rand_pred = model_randomForest.predict(X_test)
ada_pred = model_ada.predict(X_test)


# In[68]:


print_metrics(y_test, bag_pred, model_name="bagging")
print_metrics(y_test, rand_pred, model_name="randomForest")
print_metrics(y_test, ada_pred, model_name="ada")


# In[15]:


_,spending = clean_trascript(transcript)
#Fixing the type of the amount to float 
spending = spending.astype({'amount':float})

#Calculating the average spending for each user
spending_avg= spending.groupby('person').mean()
spending_avg.drop(columns = ['time'], inplace = True)

#joining them with the analytic Data Frame 
refined_df = df.join(spending_avg, on='person')
refined_df.fillna(0, inplace=True)


# In[16]:


refined_x_features = ['difficulty','duration','email','mobile','social','F', 'M','O', 'bogo', 'discount', 'informational','amount']


# In[17]:


# Split our dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(refined_df[refined_x_features], 
                                                    refined_df['success'], 
                                                    random_state=1)


# In[18]:


def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
        print('Precision score: ', format(precision_score(y_true, preds)))
        print('Recall score: ', format(recall_score(y_true, preds)))
        print('F1 score: ', format(f1_score(y_true, preds)))
        print('\n\n')
    
    else:
        print('Accuracy score for ' + model_name + ' :' , format(accuracy_score(y_true, preds)))
        print('Precision score ' + model_name + ' :', format(precision_score(y_true, preds)))
        print('Recall score ' + model_name + ' :', format(recall_score(y_true, preds)))
        print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds)))
        print('\n\n')


# In[19]:


# Fit your BaggingClassifier to the training data
model_bagging.fit(X_train,y_train)

# Fit your RandomForestClassifier to the training data
model_randomForest.fit(X_train,y_train)

# Fit your AdaBoostClassifier to the training data
model_ada.fit(X_train,y_train)


# In[20]:


# Predict using BaggingClassifier on the test data
bag_pred = model_bagging.predict(X_test)

# Predict using RandomForestClassifier on the test data
rand_pred = model_randomForest.predict(X_test)
# Predict using AdaBoostClassifier on the test data
ada_pred = model_ada.predict(X_test)


# In[21]:


# Print Bagging scores
print_metrics(y_test, bag_pred, model_name="bagging")

# Print Random Forest scores
print_metrics(y_test, rand_pred, model_name="randomForest")

# Print AdaBoost scores
print_metrics(y_test, ada_pred, model_name="ada")


# In[32]:


#model_ada = AdaBoostClassifier(n_estimators = 300, learning_rate=0.2)
model_adagd = AdaBoostClassifier()


# In[37]:


n_estimators = [250,300,350,400,450]
learning_rate = [0.05,0.1,0.2,0.3]
param_grid = dict(n_estimators = n_estimators,learning_rate = learning_rate)
grid_search = GridSearchCV(model_adagd,param_grid)
grid_result = grid_search.fit(X_train, y_train)


# In[38]:


print(grid_search.best_params_)


# In[39]:


ada_pred = grid_search.predict(X_test)
print_metrics(y_test, ada_pred, model_name="ada_grid")


# In[42]:


n_estimators = [298,300,302]
learning_rate = [0.15,0.2,0.25]
param_grid = dict(n_estimators = n_estimators,learning_rate = learning_rate)
grid_search = GridSearchCV(model_adagd,param_grid)
grid_result = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


# In[44]:


ada_pred = grid_result.predict(X_test)
print_metrics(y_test, ada_pred, model_name="ada_grid")


# In[53]:


#结论部分

## 模型的f1数值处于74~80之间，可以得出有一定帮助。目的是如何建立模型去预测优惠券的收入，方式是通过分析顾客的因素，提取出认为合适的特征，采取三种算法进行模型的训练与预测。


# ```python
# 
# ```
