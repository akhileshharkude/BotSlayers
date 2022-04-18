#!/usr/bin/env python
# coding: utf-8

# In[219]:


import numpy as np
import pandas as pd


# In[220]:


insta_data = pd.read_csv("INSTA.csv")
td = pd.read_csv("cov.csv" )


# In[221]:


td = td.drop(td.columns[0], axis=1)


# In[222]:


td.isnull().sum()


# In[223]:


td.drop(td[td['score'] > 1].index, inplace = True)


# In[224]:


td


# In[225]:


td.loc[td['category'] == "Normal"]


# In[226]:


td.loc[td["score"] < 0.50, "category"] = "Normal"


# In[227]:


td.dropna(subset = ['category'], inplace= True)


# In[228]:


td.loc[td["category"] == "Trollbot", "category"] = 1
td.loc[td["category"] == "Normal", "category"] = 0
#td.loc[td["verified"] == "False", "verified"] = 0
#td.loc[td["verified"] == "True", "verified"] = 1
#td.loc[td["geo_enabled"] == "False", "geo_enabled"] = 0
#td.loc[td["geo_enabled"] == "True", "geo_enabled"] = 1
#td.loc[td["suspended"] == False, "suspended"] = 0
#td.loc[td["suspended"] == True, "suspended"] = 1


# # Logistic Regression

# In[381]:


X = td.drop(columns=["score","joined","username","name","category","user_id","date_added","status","profile_location","listed_count","suspended","geo_enabled","verified","Unnamed: 17"],axis=1)
y = td["category"]


# In[382]:


y = y.astype('int') 
X.head()


# In[383]:


from sklearn.model_selection import train_test_split


# In[384]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=4)


# In[385]:


from sklearn.preprocessing import StandardScaler


# In[386]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[387]:


from sklearn.linear_model import LogisticRegression


# In[388]:


logmodel = LogisticRegression(random_state=4)


# In[389]:


logmodel.fit(X_train, y_train)


# In[390]:


predictions = logmodel.predict(X_test)


# In[391]:


from sklearn.metrics import classification_report


# In[392]:


classification_report(y_test,predictions)


# In[393]:


from sklearn.metrics import confusion_matrix


# In[394]:


confusion_matrix(y_test,predictions)


# In[395]:


from sklearn.metrics import accuracy_score


# In[396]:


accuracy_score(y_test,predictions)


# # Random Forest 

# In[397]:


from sklearn.ensemble import RandomForestClassifier
td=RandomForestClassifier(n_estimators=50)
td.fit(X_train, y_train)


# In[399]:


td.score(X_test,y_test)


# # Twitter API

# In[411]:


consumer_key = "FcaKOYBDGUEEEqz7ufaWIy7mp"
consumer_secret = "mihhCCjFXv2OHzAKzCvqllYNwmWfPtiztGESyTyt25O5KQx0TP"
access_token = "900365880114204672-Fs61vYMorOBrJj5rgpCZskh6fKUUDbu"
access_token_secret = "MS1NT8CNlS8I6sw2URGmhFKBAKjnEeTtMxiEYIXTFMXoQ"

b_token = "AAAAAAAAAAAAAAAAAAAAAGHWbQEAAAAAsfoFlh%2BGoBLTM6ik6bzOZTIBLwA%3D7bx9qkU0oascnv0Lr77CretCKEk7qcZ6yIHusC52vlqXfRwg6q"


# In[403]:


get_ipython().system('pip install tweepy')


# In[421]:


import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

#cursor = tweepy.Cursor(api.user_timeline, id='realDonaldTrump',tweet_mode='extended').items(1)
client = tweepy.Client(bearer_token=b_token)


id = client.get_users(usernames = ['_washermashen_'])
id
users = client.get_users_followers(id=900365880114204672)
followers = 0
for user in users:
    print(user)


# In[ ]:





# In[ ]:




