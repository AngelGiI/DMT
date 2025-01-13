import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12, 6)
import re

# Read the data
df_o = pd.read_csv('ODI-2023.csv', header=0, index_col=0, delimiter=';')
#print(df.head(3)) # print the first 3 rows of the data

# exploring raw data
print(df_o.shape) # (rows, columns)
#print(df_o.columns) # column names
#print(df_o.describe()) # summary statistics
#print(df_o.info()) # data types	and missing values

# precleaning the data
data_preclean = df_o # keeping the original data

data_preclean = data_preclean.dropna() # drop all rows with NaN values
print(data_preclean.info())

def preclean_text(data,column):
    i = 0
    for text in data[column]:  
        text = re.sub("[^A-Za-z]", " ", text) # remove all non-alphabetic characters
        text = re.sub(" +", " ", text) # remove all double spaces
        text = text.lower() # everything lowercase
        text = text.strip() # remove leading and trailing spaces
        text = text.decode() # decode to unicode
        data.ix[i,column] = text
        i += 1

def preclean_digit(data,column):
    i = 0
    for digit in data[column]:
        digit = re.sub("[^0-9]", "", digit) # remove all non-digits 
        digit = re.sub(" +", " ", digit)
        digit = digit.strip() # remove leading and trailing spaces
        data.ix[i,column] = digit
        i += 1

def string_to_float(data,column):
    i = 0
    for string in data[column]:
        string = float(string)
        data.ix[i,column] = string
        i += 1

preclean_text(data_preclean,'What programme are you in?')
preclean_text(data_preclean,'What makes a good day for you (1)?')
preclean_text(data_preclean,'What makes a good day for you (2)?')

preclean_digit(data_preclean,'How many students do you estimate there are in the room?')
preclean_digit(data_preclean,'What is your stress level (0-100)?')
preclean_digit(data_preclean,'How many hours per week do you do sports (in whole hours)? ')
preclean_digit(data_preclean,'Give a random number')

data_preclean = data_preclean.replace(r'^\s*$', np.nan, regex=True) # replace empty strings with NaN
data_preclean = data_preclean.dropna() # drop all rows with NaN values

string_to_float(data_preclean,'How many students do you estimate there are in the room?')
string_to_float(data_preclean,'What is your stress level (0-100)?')
string_to_float(data_preclean,'How many hours per week do you do sports (in whole hours)? ')
string_to_float(data_preclean,'Give a random number')

print(data_preclean.info()) # data types and missing values

'''
The code below walks through the data and counts the number of times each value appears in a column.
Descriptive statistics for the numerical columns are also printed.
For data that has been changed with precleaning, both the original and precleaned data are counted.
For every column type, plots are also created to visualize the data.
'''

# categorical columns

'''
#print(data_preclean['Have you taken a course on machine learning?'].value_counts()) # precleaned data
print(data_preclean['Have you taken a course on machine learning?'].describe()) # precleaned data
data_preclean['Have you taken a course on machine learning?'].value_counts().plot(kind='pie', title='Have you taken a course on machine learning?')
plt.show()

#print(data_preclean['Have you taken a course on information retrieval?'].value_counts()) # precleaned data
print(data_preclean['Have you taken a course on information retrieval?'].describe()) # precleaned data
data_preclean['Have you taken a course on information retrieval?'].value_counts().plot(kind='pie', title='Have you taken a course on information retrieval?')
plt.show()

#print(data_preclean['Have you taken a course on statistics?'].value_counts()) # precleaned data
print(data_preclean['Have you taken a course on statistics?'].describe()) # precleaned data
data_preclean['Have you taken a course on statistics?'].value_counts().plot(kind='pie', title='Have you taken a course on statistics?')
plt.show()

#print(data_preclean['Have you taken a course on databases?'].value_counts()) # precleaned data
print(data_preclean['Have you taken a course on databases?'].describe()) # precleaned data
data_preclean['Have you taken a course on databases?'].value_counts().plot(kind='pie', title='Have you taken a course on databases?')
plt.show()

#print(data_preclean['What is your gender?'].value_counts()) # precleaned data
print(data_preclean['What is your gender?'].describe()) # precleaned data])
data_preclean['What is your gender?'].value_counts().plot(kind='pie', title='What is your gender?')
plt.show()

#print(data_preclean['I have used ChatGPT to help me with some of my study assignments '].value_counts()) # precleaned data
print(data_preclean['I have used ChatGPT to help me with some of my study assignments '].describe()) # precleaned data
data_preclean['I have used ChatGPT to help me with some of my study assignments '].value_counts().plot(kind='pie', title='I have used ChatGPT to help me with some of my study assignments ')
plt.show()

#print(data_preclean['Did you stand up to come to your previous answer    ?'].value_counts()) # precleaned data
print(data_preclean['Did you stand up to come to your previous answer    ?'].describe()) # precleaned data
data_preclean['Did you stand up to come to your previous answer    ?'].value_counts().plot(kind='pie', title='Did you stand up to come to your previous answer?')
plt.show()
'''
'''
# medium noisy numeric columns
#print(df_o['How many students do you estimate there are in the room?'].value_counts()) # original data
#print(data_preclean['How many students do you estimate there are in the room?'].value_counts()) # precleaned data
print(data_preclean['How many students do you estimate there are in the room?'].describe())

# descriptive statistics
print('max')
print(data_preclean['How many students do you estimate there are in the room?'].max()) # max
print('min')
print(data_preclean['How many students do you estimate there are in the room?'].min()) # min
print('mean')
print(data_preclean['How many students do you estimate there are in the room?'].mean())# mean
print('median')
print(data_preclean['How many students do you estimate there are in the room?'].median())# median
print('range')
print(data_preclean['How many students do you estimate there are in the room?'].max() - data_preclean['How many students do you estimate there are in the room?'].min())# range

data_preclean['How many students do you estimate there are in the room?'].plot(kind='box', title='How many students do you estimate there are in the room?') 
plt.show()


#print(df_o['What is your stress level (0-100)?'].value_counts()) # original data
#(data_preclean['What is your stress level (0-100)?'].value_counts()) # precleaned data
print(data_preclean['What is your stress level (0-100)?'].describe())

# descriptive statistics
print('max')
print(data_preclean['What is your stress level (0-100)?'].max()) # max
print('min')
print(data_preclean['What is your stress level (0-100)?'].min()) # min
print('mean')
print(data_preclean['What is your stress level (0-100)?'].mean())# mean
print('median')
print(data_preclean['What is your stress level (0-100)?'].median())# median
print('range')
print(data_preclean['What is your stress level (0-100)?'].max() - data_preclean['What is your stress level (0-100)?'].min())# range

data_preclean['What is your stress level (0-100)?'].plot(kind='hist', title='What is your stress level (0-100)?')
plt.show()


#print(df_o['How many hours per week do you do sports (in whole hours)? '].value_counts()) # original data
#print(data_preclean['How many hours per week do you do sports (in whole hours)? '].value_counts()) # precleaned data
print(data_preclean['How many hours per week do you do sports (in whole hours)? '].describe())

# descriptive statistics
print('max')
print(data_preclean['How many hours per week do you do sports (in whole hours)? '].max()) # max
print('min')
print(data_preclean['How many hours per week do you do sports (in whole hours)? '].min()) # min
print('mean')
print(data_preclean['How many hours per week do you do sports (in whole hours)? '].mean())# mean
print('median')
print(data_preclean['How many hours per week do you do sports (in whole hours)? '].median())# median
print('range')
print(data_preclean['How many hours per week do you do sports (in whole hours)? '].max() - data_preclean['How many hours per week do you do sports (in whole hours)? '].min())# range

data_preclean['How many hours per week do you do sports (in whole hours)? '].plot(kind='hist', title='How many hours per week do you do sports (in whole hours)? ')
plt.show()

#print(df_o['Give a random number'].value_counts()) # original data
#print(data_preclean['Give a random number'].value_counts()) # precleaned data
print(data_preclean['Give a random number'].describe())

# descriptive statistics
print('max')
print(data_preclean['Give a random number'].max()) # max
print('min')
print(data_preclean['Give a random number'].min()) # min
print('mean')
print(data_preclean['Give a random number'].mean())# mean
print('median')
print(data_preclean['Give a random number'].median())# median
print('range')
print(data_preclean['Give a random number'].max() - data_preclean['Give a random number'].min())# range

data_preclean['Give a random number'].plot(kind='hist', title='Give a random number')
plt.show()

'''

'''
# very noisy numeric columns
print(data_preclean['When is your birthday (date)?'].value_counts()) # precleaned data
print(data_preclean['Time you went to bed Yesterday'].value_counts()) # precleaned data
'''

# noisy text columns
#print(df_o['What programme are you in?'].value_counts()) # original data
#print(data_preclean['What programme are you in?'].value_counts()) # precleaned data
print(data_preclean['What programme are you in?'].describe())
data_preclean['What programme are you in?'].value_counts().plot(kind='pie', title='What programme are you in?')
plt.show()

#print(df_o['What makes a good day for you (1)?'].value_counts()) # original data
#print(data_preclean['What makes a good day for you (1)?'].value_counts()) # precleaned data
print(data_preclean['What makes a good day for you (1)?'].describe())
data_preclean['What makes a good day for you (1)?'].value_counts().plot(kind='pie', title='What makes a good day for you (1)?')
#print(df_o['What makes a good day for you (2)?'].value_counts()) # original data
#print(data_preclean['What makes a good day for you (2)?'].value_counts()) # precleaned data
print(data_preclean['What makes a good day for you (2)?'].describe())
data_preclean['What makes a good day for you (2)?'].value_counts().plot(kind='pie', title='What makes a good day for you (2)?')

# cleaning the data

data = data_preclean # keeping the precleaned data

#print(data.describe())

#print(df_o.columns) # column names