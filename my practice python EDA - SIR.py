# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:48:01 2021

@author: Admin
"""

import pandas as pd

education = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/Dataset 360/education.csv")

education.workex.mean()
education.workex.median()
education.workex.mode()

from scipy import stats
stats.mode(education.workex)

# Measure of Dispersion / Second moment business decision
education.workex.var() #variance
education.workex.std() #standard variance
range = max(education.workex) - min(education.workex) #range
range

# Third moment business decision
education.workex.skew()

# Fourth moment business decision
education.workex.kurt()

# Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes
import numpy as np

plt.bar(height = education.gmat, x = np.arange(1,774,1)) # height in y axis, x- x-axis
# (1- starting from, 773(n-1), 1- step size or increment by 1)

plt.hist(education.gmat) #histogram
help(plt.hist)

plt.boxplot(education.gmat)
plt.boxplot(education.gmat,vert=False)
plt.boxplot(education.gmat,1,vert=False) #just enhancment(for understanding)

dir(plt) # directory,inside packages what are the functionalities
# folder(package) matplotlib - inside that plt(sub folder)- inside plt what are the functionality.

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(education.gmat, dist='norm',plot=pylab) #pylab is visual representation

stats.probplot(education.workex, dist='norm',plot=pylab)

#transformation to make workex variable normal
import numpy as np
stats.probplot(np.log(education.workex),dist="norm",plot=pylab) #best transformation

stats.probplot(education.workex*education.workex,dist="norm",plot=pylab) # square
stats.probplot(np.sqrt(education.workex),dist="norm",plot=pylab) # square root

# red line is IQR

# z-distribution
# cdf => cumulative distributive function; # similar to pnorm in R
# cdf(x,miu,sigma)
stats.norm.cdf(740,711,29)  # given a value, find the probability


# right side portion
1- stats.norm.cdf(740,711,29) # if it's greater than 740
# ppf => Percent point function; # similar to qnorm in R
stats.norm.ppf(0.975,0,1) # given probability, find the Z value, (95% right side, mean, sigma)
# (0.975) it's a 95% right side value, 0.025 for left side portion.

#t-distribution
#(t-value,sample size)
stats.t.cdf(1.98,139) # given a value, find the probability; # similar to pt in R
stats.t.ppf(0.975, 139) # given probability, find the t value; # similar to qt in R

education.describe()

##############################################################
###### Data Preprocessing########################################

## import packages
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

# we use ethinc diversity dataset  for this
ethnic = pd.read_csv("D:\ethnic diversity.csv")

ethnic.columns

ethnic.isna().sum()
ethnic.isnull().sum()
ethnic.describe()
ethnic.info() #object - categorical data

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ethnic.iloc[:,10:12])
df_norm.describe() # mean=0, std = 1

# or denominator (i.max()-i.min())
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min()) # or denominator (i.max()-i.min())
    return(x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ethnic.iloc[:,10:12])
df_norm.describe() # min=0, max=1

##################  creating Dummy variables using dummies ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# we use ethinc diversity dataset  for this
df = pd.read_csv("D:/ethnic diversity.csv")

df.columns
# drop emp_name column
df.drop(['Employee_Name', 'EmpID'],axis = 1, inplace=True) #axis = 1(columns),inplace = True(change in original dataset)
df.dtypes #data types

######################################
# Create dummy variables on categorcal columns

df_new = pd.get_dummies(df)

### we have created dummies for all categorical columns

#######lets us see using one hot encoding works
df1 = pd.read_csv("D:/ethnic diversity.csv")
df1.drop(['Employee_Name', 'EmpID'],axis = 1, inplace=True)
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(df1).toarray()) # 1st they convert into array then dataframe
# this one they convert all columns in dummies

# convert only categorical columns into dummies
df1 = pd.read_csv("D:/ethnic diversity.csv")
df1.drop(['EmpID','Zip', 'Salaries', 'age'],axis = 1, inplace=True) 
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(df1).toarray())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df2 = pd.read_csv("D:/ethnic diversity.csv")
df2.drop(['Employee_Name','EmpID'],axis = 1, inplace = True)
# when your target variable is non numeric- then use label encoding
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
X = df2.iloc[:,3:6] # input
y = df2['Race'] # output

df2.columns

X['Sex']= labelencoder.fit_transform(X['Sex'])
X['MaritalDesc']= labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc']= labelencoder.fit_transform(X['CitizenDesc'])

########## label encode y
y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function

# concatenate X and y

df_new = pd.concat([X,y], axis = 1)
## rename column name
df_new.columns
df_new = df_new.rename(columns={0:'Type'})

##################################################################################
###################### Outlier Treatment #########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("D:/ethnic diversity.csv")
df.dtypes
df.isna().sum()

# let's find outliers 
sns.boxplot(df.Salaries);plt.title('Boxplot');plt.show()

sns.boxplot(df.age);plt.title('Boxplot');plt.show()

# Detection of outliers 
IQR = df['Salaries'].quantile(0.75) - df['Salaries'].quantile(0.25)
lower_limit = df['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Salaries'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trimm the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_df = np.where(df['Salaries'] > upper_limit,True,np.where(df['Salaries'] < lower_limit,True,False))
# if value is greater than upper limit consider it as outliers and if the value is less than lower limit consider it as outliers
df_trimmed = df.loc[~(outliers_df),] # ~ means not - it shows all false value (not outliers)
df.shape, df_trimmed.shape # we trim 4 outliers

sns.boxplot(df_trimmed.Salaries);plt.title('Boxplot');plt.show()

#we see no outiers

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit,
                                         np.where(df['Salaries'] < lower_limit, lower_limit,
                                                  df['Salaries'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()


df['df_replaced']= pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit),
                                         np.where(df['Salaries'] < lower_limit, lower_limit), 
                                         df['Salaries']) # because of dataframe it takes 3 columns and show error




###################### 3. Winsorization #####################################
# go to anaconda powershell then type conda install -c conda-forgue feature_engine 
from feature_engine.outlier_removers import Winsorizer
windsoriser = Winsorizer(distribution='skewed', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Salaries'])
df_t = windsoriser.fit_transform(df[['Salaries']])

import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Salaries'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['Salaries']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Salaries);plt.title('Boxplot');plt.show()

###################################################################################
#################### Missing Values Imputation ##################################
import numpy as np
import pandas as pd

# load the dataset
# use modified ethnic dataset
df_raw = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/EDA,Data Types/Datasets_EDA (1)/ethnic diversity.csv")
df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/EDA,Data Types/Datasets_EDA (1)/modified ethnic.csv")

# check for count of NA'sin each column
df.isna().sum()

# There are 3 columns that have missing data ---Create an imputer object that fills 'Nan' values of SEX,MaritalDesc,Salaries
# Mean and Median imputer are used for numeric data (Salaries)
# mode is used for discrete data (SEX,MaritalDesc)

# for Mean,Meadian,Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
df["Salaries"].isnull().sum() # all 2 records replaced by mean 

# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Salaries"] = pd.DataFrame(median_imputer.fit_transform(df[['Salaries']]))
df["Salaries"].isnull().sum() # all 2 records replaced by median 

# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df[['Sex']]))
df["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df[['MaritalDesc']]))
df.isnull().sum() 

##############################################################################
################## Type casting###############################################
import pandas as pd

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/EDA,Data Types/Datasets_EDA (1)/ethnic diversity.csv")
data.dtypes

#type casting
# Now we will convert 'float64' into 'int64' type. 
data.Salaries = data.Salaries.astype('int64')
data.dtypes

#Identify duplicates records in the data
duplicate = data.duplicated()
sum(duplicate)

#Removing Duplicates
data1 = data.drop_duplicates()



