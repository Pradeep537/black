#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# A retail company wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.
# The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# ## Variable	Definition

# In[124]:


# User_ID	User ID
# Product_ID	Product ID
# Gender	Sex of User
# Age	Age in bins
# Occupation	Occupation (Masked)
# City_Category	Category of the City (A,B,C)
# Stay_In_Current_City_Years	Number of years stay in current city
# Marital_Status	Marital Status
# Product_Category_1	Product Category (Masked)
# Product_Category_2	Product may belongs to other category also (Masked)
# Product_Category_3	Product may belongs to other category also (Masked)
# Purchase	Purchase Amount (Target Variable)


# ## Importing Libraries and Loading data

# In[125]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[126]:


data = pd.read_csv("https://raw.githubusercontent.com/nanthasnk/Black-Friday-Sales-Prediction/master/Data/BlackFridaySales.csv")


# In[127]:


data.head()


# In[128]:


data.shape


# In[129]:


data.info()


# `Age` should be treated as a numerical column
# 
# `City_Category` we can convert this to a numerical column and should look at the frequency of each city category.
# 
# `Gender` has two values and should be converted to binary values
# 
# `Product_Category_2` and `Product_Category_3` have null values

# ## Checking Null values

# In[130]:


data.isnull().sum()


# ## Null Value in percentage

# In[131]:


data.isnull().sum()/data.shape[0]*100


# There are 31% null values in the `Product_Category_2` and 69% null values in the `Product_Category_3`

# # Unique elements in each attributes

# In[132]:


data.nunique()


# We can drop `User_ID` and `Product_ID` for model prediction as it has more unique values.

# # EDA

# ## Target Variable Purchase

# In[133]:


sns.distplot(data["Purchase"],color='r')
plt.title("Purchase Distribution")
plt.show()


# We can observe that purchase amount is repeating for many customers.This may be because on Black Friday many are buying discounted products in large numbers and kind of follows a Gaussian Distribution.

# In[134]:


sns.boxplot(data["Purchase"])
plt.title("Boxplot of Purchase")
plt.show()


# In[135]:


data["Purchase"].skew()


# In[136]:


data["Purchase"].kurtosis()


# In[137]:


data["Purchase"].describe()


# The purchase is right skewed and we can observe multiple peaks in the distribution we can do a log transformation for the purchase.

# ### Gender

# In[138]:


sns.countplot(data['Gender'])
plt.show()


# In[139]:


data['Gender'].value_counts(normalize=True)*100


# There are more males than females

# In[140]:


data.groupby("Gender").mean()["Purchase"]


# On average the male gender spends more money on purchase contrary to female, and it is possible to also observe this trend by adding the total value of purchase.

# ### Marital Status

# In[141]:


sns.countplot(data['Marital_Status'])
plt.show()


# There are more unmarried people in the dataset who purchase more

# In[142]:


data.groupby("Marital_Status").mean()["Purchase"]


# In[143]:


data.groupby("Marital_Status").mean()["Purchase"].plot(kind='bar')
plt.title("Marital_Status and Purchase Analysis")
plt.show()


# This is interesting though unmarried people spend more on purchasing, the average purchase amount of married and unmarried people are the same.

# ### Occupation

# In[144]:


plt.figure(figsize=(18,5))
sns.countplot(data['Occupation'])
plt.show()


# Occupation has at least 20 different values. Since we do not known to each occupation each number corresponds, is difficult to make any analysis. Furthermore, it seems we have no alternative but to use since there is no way to reduce this number

# In[145]:


occup = pd.DataFrame(data.groupby("Occupation").mean()["Purchase"])
occup


# In[146]:


occup.plot(kind='bar',figsize=(15,5))
plt.title("Occupation and Purchase Analysis")
plt.show()


# Although there are some occupations which have higher representations, it seems that the amount each user spends on average is more or less the same for all occupations. Of course, in the end, occupations with the highest representations will have the highest amounts of purchases.

# ### City_Category

# In[147]:


sns.countplot(data['City_Category'])
plt.show()


# It is observed that city category B has made the most number of puchases.

# In[148]:


data.groupby("City_Category").mean()["Purchase"].plot(kind='bar')
plt.title("City Category and Purchase Analysis")
plt.show()


# However, the city whose buyers spend the most is city type ‘C’.

# ### Stay_In_Current_City_Years

# In[149]:


sns.countplot(data['Stay_In_Current_City_Years'])
plt.show()


# It looks like the longest someone is living in that city the less prone they are to buy new things. Hence, if someone is new in town and needs a great number of new things for their house that they’ll take advantage of the low prices in Black Friday to purchase all the things needed.

# In[151]:


data.groupby("Stay_In_Current_City_Years").mean()["Purchase"].plot(kind='bar')
plt.title("Stay_In_Current_City_Years and Purchase Analysis")
plt.show()


# We see the same pattern seen before which show that on average people tend to spend the same amount on purchases regardeless of their group. People who are new in city are responsible for the higher number of purchase, however looking at it individually they tend to spend the same amount independently of how many years the have lived in their current city.

# ### Age

# In[152]:


sns.countplot(data['Age'])
plt.title('Distribution of Age')
plt.xlabel('Different Categories of Age')
plt.show()


# Age 26-35 Age group makes the most no of purchases in the age group.

# In[153]:


data.groupby("Age").mean()["Purchase"].plot(kind='bar')


# Mean puchase rate between the age groups tends to be the same except that the 51-55 age group has a little higher average purchase amount

# In[154]:


data.groupby("Age").sum()['Purchase'].plot(kind="bar")
plt.title("Age and Purchase Analysis")
plt.show()


# Total amount spent in purchase is in accordance with the number of purchases made, distributed by age.

# ### Product_Category_1

# In[155]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_1'])
plt.show()


# It is clear that `Product_Category_1` numbers 1,5 and 8 stand out. Unfortunately we don't know which product each number represents as it is masked.

# In[156]:


data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Mean Analysis")
plt.show()


# If you see the value spent on average for Product_Category_1 you see that although there were more products bought for categories 1,5,8 the average amount spent for those three is not the highest. It is interesting to see other categories appearing with high purchase values despite having low impact on sales number.

# In[157]:


data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Analysis")
plt.show()


# The distribution that we saw for this predictor previously appears here. For example, those three products have the highest sum of sales since their were three most sold products.

# ### Product_Category_2

# In[158]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_2'])
plt.show()


# ### Product_Category_3

# In[159]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_3'])
plt.show()


# In[160]:


data.corr()


# ## HeatMap

# In[161]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# There is a some corellation between the product category groups.

# In[162]:


data.columns


# In[163]:


df = data.copy()


# In[164]:


df.head()


# In[165]:


# df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace(to_replace="4+",value="4")


# In[166]:


#Dummy Variables:
df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])


# ## Encoding the categorical variables

# In[167]:


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()


# In[168]:


df['Gender'] = lr.fit_transform(df['Gender'])


# In[169]:


df['Age'] = lr.fit_transform(df['Age'])


# In[170]:


df['City_Category'] = lr.fit_transform(df['City_Category'])


# In[171]:


df.head()


# In[173]:


df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')


# In[174]:


df.isnull().sum()


# In[175]:


df.info()


# ## Dropping the irrelevant columns

# In[176]:


df = df.drop(["User_ID","Product_ID"],axis=1)


# ## Splitting data into independent and dependent variables

# In[179]:


X = df.drop("Purchase",axis=1)


# In[180]:


y=df['Purchase']


# In[181]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# ## Modeling

# ### Linear Regression

# In[182]:


from sklearn.linear_model import LinearRegression


# In[183]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[184]:


lr.intercept_


# In[185]:


lr.coef_


# In[186]:


y_pred = lr.predict(X_test)


# In[187]:


from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


# In[188]:


mean_absolute_error(y_test, y_pred)


# In[189]:


mean_squared_error(y_test, y_pred)


# In[190]:


r2_score(y_test, y_pred)


# In[191]:


from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, y_pred)))


# # DecisionTreeRegressor

# In[192]:


from sklearn.tree import DecisionTreeRegressor

# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  


# In[193]:


regressor.fit(X_train, y_train)


# In[194]:


dt_y_pred = regressor.predict(X_test)


# In[195]:


mean_absolute_error(y_test, dt_y_pred)


# In[196]:


mean_squared_error(y_test, dt_y_pred)


# In[197]:


r2_score(y_test, dt_y_pred)


# In[198]:


from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, dt_y_pred)))


# # Random Forest Regressor

# In[199]:


from sklearn.ensemble import RandomForestRegressor

# create a regressor object 
RFregressor = RandomForestRegressor(random_state = 0)  


# In[200]:


RFregressor.fit(X_train, y_train)


# In[201]:


rf_y_pred = RFregressor.predict(X_test)


# In[202]:


mean_absolute_error(y_test, rf_y_pred)


# In[203]:


mean_squared_error(y_test, rf_y_pred)


# In[204]:


r2_score(y_test, rf_y_pred)


# In[205]:


from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, rf_y_pred)))


# # XGBoost Regressor

# In[206]:


from xgboost.sklearn import XGBRegressor


# In[207]:


xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(X_train, y_train)


# In[208]:


xgb_y_pred = xgb_reg.predict(X_test)


# In[209]:


mean_absolute_error(y_test, xgb_y_pred)


# In[210]:


mean_squared_error(y_test, xgb_y_pred)


# In[211]:


r2_score(y_test, xgb_y_pred)


# In[212]:


from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, xgb_y_pred)))


# The ML algorithm that perform the best was XGBoost Regressor Model with RMSE = 2879
