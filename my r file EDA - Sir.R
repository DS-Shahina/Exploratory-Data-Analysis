2 + 2 # control + enter or control +R
# works as calculator

#Packages
install.packages("readr")
library(readr)

#Read data into R
education <- read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/Dataset 360")
education <- read.csv(file.choose())

View(education)

#Exploratory Data Analysis
#Measures of Central Tendency / First moment business decision

mean(education$workex) # '$' is used to refer to the variables within object
attach(education) # When used we can directly refer to the variable name

mean(gmat)
mean(workex)

rm(education) #Remove specific object to free RAM (memory)
rm(list = ls()) # Remove all to free RAM (memory)

median(workex)

mode(workex) # they show data type

#Mode
y <- c(19, 4, 5, 7, 29, 19,19,19,19, 29, 13, 25,29,5,5,5,5,4,4,4,4)
# uni mode function
Mode <- function(x){
     a = unique(x) # x is a vector
     return(a[which.max(tabulae(match(x,a)))])
}

mode(y)
# Bi mode function
modes <- function(x) {
  ux <- unique(x)
  tab <- tabulate(match(x, ux))
  ux[tab == max(tab)]
}
modes(y)

# Measures of Dispersion / Second moment business decision
var(workex) # variance
sd(workex) # standard deviation
range <- max(workex) - min(workex) #range
range

install.packages("moments")
library(moments)

#Third moment business decision
skewness(workex)

#Fourth moment business decision
kurtosis(workex)

#Graphical Representation
barplot(gmat)
dotchart(gmat) 
# it's a left skew data, but mean is greater than median 
# but generally for left skew data mean should be less than median
# why because most of the data lie on the left side and very less ouliers 
# so mean is not that much influenced by outliers

mean(gmat)
median(gmat)

hist(gmat) # histogram

boxplot(gmat)
y <- boxplot(gmat)
y$out # to see outliers

#Probability Distribution
install.packages("UsingR")
library("UsingR")
densityplot(gmat)

# Normal Quantile-Quantile Plot
qqnorm(gmat)
qqline(gmat)
qqnorm(workex)
qqline(workex)

# Transformation to make workex variable normal
qqnorm(log(workex))
qqline(log(workex))

# Data Pre-Processing
# use ethnic diversity dataset
# packages such as 'dummyvars', 'fastdummies' can be used 
# loading the dataset
data <- read.csv(file.choose())

# Checking str and summary of the data
str(data) # data type
summary(data)
attach(data)

install.packages("fastDummies")
library(fastDummies)

# One-hot-encoding - coverting categorical into numeric
# if you want to treat your non numeric data as nominal data then go one hot encoding

data_dummy <- dummy_cols(data, select_columns = c("Position","State","Sex","MaritalDesc","CitizenDesc","EmploymentStatus","Department","Race"),
                         remove_first_dummy = TRUE,remove_most_frequent_dummy = FALSE, remove_selected_columns = TRUE)


??fastDummies
# you can choose between remove_first_dummy and remove_most_frequent_dummy
# as both will remove multicollinearity issue 
################################################
###### Normalization ###################################
# we will ethnic diversity dataset
# when use normalization?
#if data has 0 & 1 value more or dummy variable then use normalization

df <- data_dummy[, -c(1,2,3)]

# to normalize the data we use custom function 
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

df_norm <- as.data.frame(lapply(df, norm)) #first convert 
#column into list and apply normalization then again convert into dataframe or columns.

summary(df_norm) #for checking

# To apply standardization we have inbuilt function scale
# we use mtcars dataset
df_mt <- read.csv(file.choose())
View(df_mt)
#remove first column
df_mt <- df_mt[,-1]
# use scale function
df_scale <- as.data.frame(scale(df_mt))
# hypothetically you have values ranging between -3 to +3 or in general the range is -inf to +inf

summary(df_scale)

#################################################
### Label encoding in R - converting into ordinal data
# if your data is ordinal then go with label encoding
library(CatEncoders)
View(data)

# character column: 'Position'
lb_new <- LabelEncoder.fit(data$Position)
lb_new # indexing according to alphabetical order
# new values are transformed to NA
position_new <- transform(lb_new,data$Position)
position_new

# Using cbind to combine with original dataset
newdata <- cbind(data, position_new)
newdata

# Handling duplicates###############################################
# Duplicate entries are removed using 'duplicated' function
# it stores the duplicate values into another variable

dup <- duplicated(data)
dup
data_new <- data[!duplicated(data),]#checking rows if duplicate is there or not
data_new

####### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ####################
###################### Type Casting ########################
# Type casting is used to convert one data type into another.
# ex: categorical data to numeric data

age_num <- as.numeric(data$age) #int to numeric(decimal)
age_num

is.integer(data$age) #it will return true or false
is.character(data$age)

# Ex: Convert Numeric to Integer
data$age <- as.integer(data$age)

# Ex: Convert categorical data to factor type
str(data) # categorical data is character

# convert sex column to factors
data$Sex <- as.factor(data$Sex)
str(data) # check now for sex column

# Alternatively we can use argument stringAsFactors=TRUE
# load ethnic diversity dataset
data1 <- read.csv(file.choose(),stringsAsFactors = TRUE)
str(data1)
summary(data1)


################################################
## Missing values - imputation

# Lets introduce NA values into the dataset using 'missForest' package

install.packages("missForest")
library(missForest)

# Generate 10% missing values at Random
# remove three unwanted columns as they are not required for analysis
data.mis <- prodNA(data, noNA = 0.1)
summary(data.mis)
attach(data.mis)

# Remove categorical variables
data.mis <- subset(data.mis, select = -c(Employee_Name, EmpID, Position, State, Sex, MaritalDesc,CitizenDesc,
                                         EmploymentStatus,Department,Race))


summary(data.mis)

# MICE (Multivariate Imputation via chained Equations)
install.packages("mice")
library(mice)
# MICE assumes that the missing data are Missing at Random(MAR)
# Means that the probability that a value is missing depends only



md.pattern(data.mis[-1])

impute_data <- mice(data.mis, m=5, maxit=50, method='pmm',
                    seed = 500)
summary(impute_data) #Predictive mean matching(pmm)

# m - Refers to 5 imputed data sets
# maxit - Refers to no. of iterations taken to impute missing value
# method - Refers to method used in imputation
# we used predictive mean matching
#check imputation has happened or not

impute_data$imp$Salaries # check the imputed values
# we have packages such as Amelia, missForest to apply

# Outlier Treatment
View(data)
boxplot(data$Salaries)
boxplot(data$age)


# There are 2 outliers in the salary column
#Replace outliers with the maximum value - Winsorization
qunt1 <- quantile(data$Salaries,probs = c(.25,.75))
qunt1 # 25% 23092, # 75% = 51452
caps <- quantile(data$Salaries,probs = c(.01,.99), na.rm = T)
caps # 1 % = 441, 99% = 95770
H <- 1.5*IQR( data$Salaries, na.rm = T)
H #42539.92
data$Salaries[data$Salaries<(qunt1[1]-H)] <- caps[1]
data$Salaries[data$Salaries>(qunt1[2]+H)] <- caps[2]
boxplot(data$Salaries)

####################################################
############# zero Variance
library(ggplot2)
library(ggthemes)
# Use 'apply' and 'var' functions to
# check for variance on numerical values
apply(data, 2, var)

# Check for columns having zero variance
 which(apply(data, 2, var)==0) # ignore the warnings

 
###################
# z-distribution
#pnorm(x,miu,sigma)
pnorm(680,711,29) # given a value, find the probability, pnorm is probability of a normal distributed data
qnorm(0.025) # given probability, find the Z value , # for 95%, left side portion
#(0.975) it's a 95% right side value, 0.025 for left side portion.
 
 
# t-distribution
#pt(t-value,sample size)
#sample size (n) = 140, n-1 = 139
#degree of freedom = 139
pt(1.98, 139) # given a value, find the probability
qt(0.975, 139) # given probability, find the t value,

