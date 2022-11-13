#!/usr/bin/env python
# coding: utf-8

# ## Bank Telemarketing Campaign Case Study.

# In this case study you’ll be learning Exploratory Data Analytics with the help of a case study on "Bank marketing campaign". This will enable you to understand why EDA is a most important step in the process of Machine Learning.

# #### Problem Statement:

#  
# 
# The bank provides financial services/products such as savings accounts, current accounts, debit cards, etc. to its customers. In order to increase its overall revenue, the bank conducts various marketing campaigns for its financial products such as credit cards, term deposits, loans, etc. These campaigns are intended for the bank’s existing customers. However, the marketing campaigns need to be cost-efficient so that the bank not only increases their overall revenues but also the total profit. You need to apply your knowledge of EDA on the given dataset to analyse the patterns and provide inferences/solutions for the future marketing campaign.
# 
# The bank conducted a telemarketing campaign for one of its financial products ‘Term Deposits’ to help foster long-term relationships with existing customers. The dataset contains information about all the customers who were contacted during a particular year to open term deposit accounts.
# 
# 
# **What is the term Deposit?**
# 
# Term deposits also called fixed deposits, are the cash investments made for a specific time period ranging from 1 month to 5 years for predetermined fixed interest rates. The fixed interest rates offered for term deposits are higher than the regular interest rates for savings accounts. The customers receive the total amount (investment plus the interest) at the end of the maturity period. Also, the money can only be withdrawn at the end of the maturity period. Withdrawing money before that will result in an added penalty associated, and the customer will not receive any interest returns.
# 
# Your target is to do end to end EDA on this bank telemarketing campaign data set to infer knowledge that where bank has to put more effort to improve it's positive response rate. 

# #### Importing the libraries.

# In[2]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[3]:


#import the useful libraries.
import numpy as np , pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns


# ## Session- 2, Data Cleaning 

# ### Segment- 2, Data Types 

# There are multiple types of data types available in the data set. some of them are numerical type and some of categorical type. You are required to get the idea about the data types after reading the data frame. 
# 
# Following are the some of the types of variables:
# - **Numeric data type**: banking dataset: salary, balance, duration and age.
# - **Categorical data type**: banking dataset: education, job, marital, poutcome and month etc.
# - **Ordinal data type**: banking dataset: Age group.
# - **Time and date type** 
# - **Coordinates type of data**: latitude and longitude type.
# 

# #### Read in the Data set. 

# In[4]:


#read the data set of "bank telemarketing campaign" in inp0.
inp0=pd.read_csv('V1.csv')


# In[5]:


#Print the head of the data frame.
inp0.head()


# ### Segment- 3, Fixing the Rows and Columns 

# Checklist for fixing rows:
# - **Delete summary rows**: Total and Subtotal rows
# - **Delete incorrect rows**: Header row and footer row
# - **Delete extra rows**: Column number, indicators, Blank rows, Page No.
# 
# Checklist for fixing columns:
# - **Merge columns for creating unique identifiers**, if needed, for example, merge the columns State and City into the column Full address.
# - **Split columns to get more data**: Split the Address column to get State and City columns to analyse each separately. 
# - **Add column names**: Add column names if missing.
# - **Rename columns consistently**: Abbreviations, encoded columns.
# - **Delete columns**: Delete unnecessary columns.
# - **Align misaligned columns**: The data set may have shifted columns, which you need to align correctly.
# 

# #### Read the file without unnecessary headers.

# In[6]:


#read the file in inp0 without first two rows as it is of no use.
inp0=pd.read_csv('V1.csv',skiprows=2)


# In[7]:


#print the head of the data frame.
inp0.head()


# In[8]:


#print the information of variables to check their data types.
inp0.info()


# In[9]:


#convert the age variable data type from float to integer.


# In[10]:


#print the average age of customers.


# #### Dropping customer id column. 

# In[11]:


#drop the customer id as it is of no use.
inp0.drop('customerid',axis=1,inplace=True)
inp0.head()


# #### Dividing "jobedu" column into job and education categories. 

# In[12]:


#Extract job in newly created 'job' column from "jobedu" column.
inp0['job']=inp0.jobedu.apply(lambda x : x.split(',')[0])
inp0.head()


# In[13]:


#Extract education in newly created 'education' column from "jobedu" column.
inp0['education']=inp0.jobedu.apply(lambda x : x.split(',')[1])


# In[14]:


#drop the "jobedu" column from the dataframe.
inp0.head()


# In[ ]:





# ### Segment- 4, Impute/Remove missing values 

# Take aways from the lecture on missing values:
# 
# - **Set values as missing values**: Identify values that indicate missing data, for example, treat blank strings, "NA", "XX", "999", etc., as missing.
# - **Adding is good, exaggerating is bad**: You should try to get information from reliable external sources as much as possible, but if you can’t, then it is better to retain missing values rather than exaggerating the existing rows/columns.
# - **Delete rows and columns**: Rows can be deleted if the number of missing values is insignificant, as this would not impact the overall analysis results. Columns can be removed if the missing values are quite significant in number.
# - **Fill partial missing values using business judgement**: Such values include missing time zone, century, etc. These values can be identified easily.
# 
# Types of missing values:
# - **MCAR**: It stands for Missing completely at random (the reason behind the missing value is not dependent on any other feature).
# - **MAR**: It stands for Missing at random (the reason behind the missing value may be associated with some other features).
# - **MNAR**: It stands for Missing not at random (there is a specific reason behind the missing value).
# 

# #### handling missing values in age column.

# In[15]:


#count the missing values in age column.
inp0.isnull().sum()


# In[16]:


#pring the shape of dataframe inp0
inp0.shape


# In[17]:


#calculate the percentage of missing values in age column.
100.0*(20/len(inp0))


# Drop the records with age missing. 

# In[18]:


#drop the records with age missing in inp0 and copy in inp1 dataframe.
inp1=inp0[~(inp0['age'].isnull())]
inp1.head()


# #### handling missing values in month column

# In[19]:


#count the missing values in month column in inp1.
inp1.month.isnull().sum()


# In[20]:


#print the percentage of each month in the data frame inp1.
100.0*(inp1.month.isnull().sum()/len(inp1))


# In[21]:


#find the mode of month in inp1
inp1.month.mode()[0]


# In[22]:


# fill the missing values with mode value of month in inp1.
inp1.month.fillna(inp1.month.mode()[0],inplace=True)


# In[23]:


#let's see the null values in the month column.
inp1.month.isnull().sum()


# #### handling missing values in response column 

# In[24]:


#count the missing values in response column in inp1.
inp1.response.isnull().sum()


# In[25]:


#calculate the percentage of missing values in response column. 
100.0*(inp1.response.isnull().sum()/len(inp1))


# Target variable is better of not imputed.
# - Drop the records with missing values.

# In[26]:


#drop the records with response missings in inp1.
inp2=inp1[~(inp1.response.isnull())]
inp2.head()


# In[27]:


#calculate the missing values in each column of data frame: inp1.
inp2.isnull().sum()


# #### handling pdays column. 

# In[28]:


#describe the pdays column of inp1.
inp2.pdays.describe()


# -1 indicates the missing values.
# Missing value does not always be present as null.
# How to handle it:
# 
# Objective is:
# - you should ignore the missing values in the calculations
# - simply make it missing - replace -1 with NaN.
# - all summary statistics- mean, median etc. we will ignore the missing values of pdays.

# In[29]:


#describe the pdays column with considering the -1 values.
inp2.loc[inp2.pdays<0,'pdays']=np.NaN
inp2.pdays.describe()


# ### Segment- 5, Handling Outliers 

# Major approaches to the treat outliers:
#  		
# - **Imputation**
# - **Deletion of outliers**
# - **Binning of values**
# - **Cap the outlier**
# 

# #### Age variable 

# In[30]:


#describe the age variable in inp1.
inp2.age.describe()


# In[31]:


#plot the histogram of age variable.
inp2.age.plot.hist()
plt.show()


# In[32]:


#plot the boxplot of age variable.
sns.boxplot(inp2.age)
plt.show()


# #### Salary variable 

# In[33]:


#describe the salary variable of inp1.
inp2.salary.describe()


# In[34]:


#plot the boxplot of salary variable.
sns.boxplot(inp2.salary)


# #### Balance variable 

# In[35]:


#describe the balance variable of inp1.
inp2.balance.describe()


# In[36]:


#plot the boxplot of balance variable.
sns.boxplot(inp2.balance)


# In[37]:


#plot the boxplot of balance variable after scaling in 8:2.
plt.figure(figsize=[8,2])
sns.boxplot((inp2.balance))
plt.show()


# In[38]:


#print the quantile (0.5, 0.7, 0.9, 0.95 and 0.99) of balance variable
inp2.balance.quantile ([0.5, 0.7, 0.9, 0.95,0.99])


# ### Segment- 6, Standardising values 

# Checklist for data standardization exercises:
# - **Standardise units**: Ensure all observations under one variable are expressed in a common and consistent unit, e.g., convert lbs to kg, miles/hr to km/hr, etc.
# - **Scale values if required**: Make sure all the observations under one variable have a common scale.
# - **Standardise precision** for better presentation of data, e.g., change 4.5312341 kg to 4.53 kg.
# - **Remove extra characters** such as common prefixes/suffixes, leading/trailing/multiple spaces, etc. These are irrelevant to analysis.
# - **Standardise case**: String variables may take various casing styles, e.g., UPPERCASE, lowercase, Title Case, Sentence case, etc.
# - **Standardise format**: It is important to standardise the format of other elements such as date, name, etce.g., change 23/10/16 to 2016/10/23, “Modi, Narendra” to “Narendra Modi", etc.

# #### Duration variable

# In[39]:


#describe the duration variable of inp1
inp2.duration.describe()


# In[40]:


#convert the duration variable into single unit i.e. minutes. and remove the sec or min prefix.
inp2.duration= inp2.duration.apply(lambda x : float(x.split()[0])/60 if x.find("sec")>0 else float(x.split()[0]))
inp2.head()


# In[41]:


#describe the duration variable
inp2.duration.describe()


# ## Session- 3, Univariate Analysis 

# ### Segment- 2, Categorical unordered univariate analysis 

# Unordered data do not have the notion of high-low, more-less etc. Example:
# - Type of loan taken by a person = home, personal, auto etc.
# - Organisation of a person = Sales, marketing, HR etc.
# - Job category of persone.
# - Marital status of any one.
# 

# #### Marital status 

# In[42]:


#calculate the percentage of each marital status category. 
Ms=100*(inp2.marital.value_counts())/len(inp2)


# In[43]:


#plot the bar graph of percentage marital status categories
inp2.marital.value_counts(normalize=True).plot.barh()
plt.show()


# #### Job  

# In[44]:


#calculate the percentage of each job status category.
jb=100*(inp2.job.value_counts())/len(inp2)
jb


# In[45]:


#plot the bar graph of percentage job categories
inp2.job.value_counts(normalize=True).plot.barh()
plt.show()


# ### Segment- 3, Categorical ordered univariate analysis 

# Ordered variables have some kind of ordering. Some examples of bank marketing dataset are:
# - Age group= <30, 30-40, 40-50 and so on.
# - Month = Jan-Feb-Mar etc.
# - Education = primary, secondary and so on.

# #### Education

# In[46]:


#calculate the percentage of each education category.
ED= 100 *(inp2.education.value_counts())/len(inp2)
ED


# In[47]:


#plot the pie chart of education categories
inp2.education.value_counts(normalize=True).plot.pie()
plt.show()


# #### poutcome 

# In[48]:


#calculate the percentage of each poutcome category.
Pc= 100*(inp2.poutcome.value_counts())/len(inp2)
Pc


# #### Response the target variable 

# In[49]:


#calculate the percentage of each response category.
Pc= 100*(inp2.response.value_counts())/len(inp2)
Pc


# In[50]:


#plot the pie chart of response categories
inp2.response.value_counts(normalize=True).plot.pie()
plt.show()


# ## Session- 4, Bivariate and Multivariate Analysis

# ### Segment-2, Numeric- numeric analysis 

# There are three ways to analyse the numeric- numeric data types simultaneously.
# - **Scatter plot**: describes the pattern that how one variable is varying with other variable.
# - **Correlation matrix**: to describe the linearity of two numeric variables.
# - **Pair plot**: group of scatter plots of all numeric variables in the data frame.

# In[51]:


#plot the scatter plot of balance and salary variable in inp1
plt.scatter(inp2.balance,inp2.salary)
plt.show()


# In[52]:


#plot the scatter plot of balance and age variable in inp1
inp2.plot.scatter(x='age',y='balance')
plt.show()


# In[53]:


#plot the pair plot of salary, balance and age in inp1 dataframe.
sns.pairplot(data=inp2,vars=['salary','balance','age'])
plt.show()


# #### Correlation heat map 

# In[54]:


#plot the correlation matrix of salary, balance and age in inp1 dataframe.
sns.heatmap(inp2[['age','salary','balance']].corr(),annot=True ,cmap='Reds')


# ### Segment- 4, Numerical categorical variable

# #### Salary vs response 

# In[55]:


#groupby the response to find the mean of the salary with response no & yes seperatly.
inp2.groupby('response')['salary'].mean()


# In[56]:


#groupby the response to find the median of the salary with response no & yes seperatly.
inp2.groupby('response')['salary'].median()


# In[57]:


#plot the box plot of salary for yes & no responses.
sns.boxplot(data=inp2,x='response',y='salary')
plt.show()


# #### Balance vs response 

# In[59]:


#plot the box plot of balance for yes & no responses.
sns.boxplot(data=inp2,x='balance',y='response')
plt.show()


# In[61]:


#groupby the response to find the mean of the balance with response no & yes seperatly.
inp2.groupby('response')['balance'].mean()


# In[62]:


#groupby the response to find the median of the balance with response no & yes seperatly.
inp2.groupby('response')['balance'].median()


# ##### 75th percentile 

# In[64]:


#function to find the 75th percentile.
def p75(x):
    return np.quantile(x, 0.75)


# In[66]:


#calculate the mean, median and 75th percentile of balance with response
inp1.groupby('response')['balance'].aggregate(['mean','median',p75])


# In[83]:


#plot the bar graph of balance's mean an median with response.
inp1.groupby('response')['balance'].aggregate(['mean','median']).plot.bar()
plt.show()


# #### Education vs salary 

# In[68]:


#groupby the education to find the mean of the salary education category.
inp2.groupby('education')['salary'].mean()


# In[69]:


#groupby the education to find the median of the salary for each education category.
inp2.groupby('education')['salary'].median()


# In[84]:


inp2.groupby('education')['salary'].aggregate(['mean','median']).plot.bar()
plt.show()


# #### Job vs salary

# In[85]:


#groupby the job to find the mean of the salary for each job category.
inp2.groupby('job')['salary'].mean().plot.bar()
plt.show()


# ### Segment- 5, Categorical categorical variable 

# In[72]:


#create response_flag of numerical data type where response "yes"= 1, "no"= 0
inp2['response_flag']=np.where(inp2.response=='yes',1,0)


# In[74]:


inp2.response_flag.value_counts()


# #### Education vs response rate

# In[78]:


#calculate the mean of response_flag with different education categories.
inp2.groupby('education')['response_flag'].mean()


# In[86]:


inp2.groupby('education')['response_flag'].mean().plot.bar()
plt.show()


# #### Marital vs response rate 

# In[88]:


#calculate the mean of response_flag with different marital status categories.
inp2.groupby('marital')['response_flag'].mean()


# In[89]:


#plot the bar graph of marital status with average value of response_flag
inp2.groupby('marital')['response_flag'].mean().plot.barh()
plt.show()


# #### Loans vs response rate 

# In[90]:


#plot the bar graph of personal loan status with average value of response_flag
inp2.groupby('loan')['response_flag'].mean()


# In[91]:


inp2.groupby('loan')['response_flag'].mean().plot.bar()


# #### Housing loans vs response rate 

# In[92]:


#plot the bar graph of housing loan status with average value of response_flag
inp2.groupby('housing')['response_flag'].mean()


# In[96]:


inp2.groupby('housing')['response_flag'].mean().plot.bar()
plt.show()


# #### Age vs response 

# In[97]:


#plot the boxplot of age with response_flag
inp2.groupby('age')['response_flag'].mean().plot.box()
plt.show()


# ##### making buckets from age columns 

# In[103]:


#create the buckets of <30, 30-40, 40-50 50-60 and 60+ from age column.
pd.cut(inp2.age[:5],[0,30,40,50,60,9999],labels=['<30','30-40','40-50','50-60','60+'])


# In[122]:


inp2['age_group']=pd.cut(inp2.age,[0,30,40,50,60,9999],labels=['<30','30-40','40-50','50-60','60+'])


# In[123]:


inp2.age_group.value_counts(normalize=True)


# In[124]:


#plot the percentage of each buckets and average values of response_flag in each buckets. plot in subplots.
plt.figure(figsize=[10,4])
plt.subplot(1,2,1)
inp2.age_group.value_counts(normalize=True).plot.bar()
plt.subplot(1,2,2)
inp2.groupby(['age_group'])['response_flag'].mean().plot.bar()
plt.show()


# In[125]:


#plot the bar graph of job categories with response_flag mean value.
plt.figure(figsize=[10,4])
plt.subplot(1,2,1)
inp2.age_group.value_counts(normalize=True).plot.bar()
plt.subplot(1,2,2)
inp2.groupby(['job'])['response_flag'].mean().plot.bar()
plt.show()


# ### Segment-6, Multivariate analysis 

# #### Education vs marital vs response 

# In[128]:


#create heat map of education vs marital vs response_flag
res=pd.pivot_table(data=inp2,index='education',columns='marital',values='response_flag')
res


# In[135]:


sns.heatmap(res,annot=True,cmap='rainbow',center=0.117)
plt.show()


# #### Job vs marital vs response 

# In[136]:


#create the heat map of Job vs marital vs response_flag.
res1=pd.pivot_table(data=inp2,index='job',columns='marital',values='response_flag')
res1


# In[144]:


sns.heatmap(res1,annot=True,cmap='RdYlGn',center=0.117)
plt.show()


# #### Education vs poutcome vs response

# In[139]:


#create the heat map of education vs poutcome vs response_flag.
res2=pd.pivot_table(data=inp2,index='education',columns='poutcome',values='response_flag')
res2


# In[142]:


sns.heatmap(res2,annot=True,cmap='RdYlGn_r',center=0.117)
plt.show()

