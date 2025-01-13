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

#print(data_preclean.info())

def preclean_text(data,column):
    i = 0
    for text in data[column]: 
        if isinstance(text, str): # check if string
            text = re.sub("[^A-Za-z]", " ", text) # remove all non-alphabetic characters
            text = re.sub(" +", " ", text) # remove all double spaces
            text = text.lower() # everything lowercase
            text = text.strip() # remove leading and trailing spaces

        elif isinstance(text, bytes):
            text = text.decode() # decode to unicode when byte string

        data.loc[i,column] = text
        i += 1

def preclean_digit(data,column):
    i = 0
    for digit in data[column]:
        if isinstance(digit, str) and column == 'What is your stress level (0-100)?' : # check if string
            digit = re.sub("[^0-9]", "", digit) # remove all non-digits
            if i >300 :
                print('hehe')
                print(digit) 
            digit = re.sub(" +", " ", digit)
            if i >300 :
                print('hihi')
                print(digit)            
            digit = digit.strip() # remove leading and trailing spaces
            if i >300 :
                print('hoho')
                print(digit) 
            data_preclean.loc[i,column] = digit
            if i >300 :
                print(data[column][i])
                print(data_preclean[column][i])
            i += 1
        elif column == 'What is your stress level (0-100)?' : # check if string
            print('wtf')

def string_to_float(data,column):
    i = 0
    for string in data[column]:
        if isinstance(string, str): # check if string
            string = float(string)
            data.loc[i,column] = string
            i += 1

def stress_level(data,column):
    i = 0
    for level in data[column]:
        level = re.sub("[^0-9]", "", level) # remove all non-digits 
        level = re.sub(" +", " ", level)
        level = level.strip() # remove leading and trailing spaces
        data.loc[i,column] = level
        i += 1

preclean_text(data_preclean,'What programme are you in?')
preclean_text(data_preclean,'What makes a good day for you (1)?')
preclean_text(data_preclean,'What makes a good day for you (2)?')

preclean_digit(data_preclean,'How many students do you estimate there are in the room?')
preclean_digit(data_preclean,'What is your stress level (0-100)?')
preclean_digit(data_preclean,'How many hours per week do you do sports (in whole hours)? ')
preclean_digit(data_preclean,'Give a random number')


print(data_preclean['What is your stress level (0-100)?'][300:303]) # (rows, columns)
data_preclean = data_preclean.replace(r'^\s*$', np.nan, regex=True) # replace empty strings with NaN
print(data_preclean['What is your stress level (0-100)?'][300:303]) # (rows, columns)
data_preclean = data_preclean.dropna() # drop all rows with NaN values
print(data_preclean.shape) # (rows, columns)

# string_to_float(data_preclean,'How many students do you estimate there are in the room?')
# string_to_float(data_preclean,'What is your stress level (0-100)?')
# string_to_float(data_preclean,'How many hours per week do you do sports (in whole hours)? ')
# string_to_float(data_preclean,'Give a random number')

#stress_level(data_preclean,'What is your stress level (0-100)?')

#print(df_o['What is your stress level (0-100)?'].value_counts()) # original data
print(data_preclean['What is your stress level (0-100)?'].value_counts()) # precleaned data
#print(data_preclean['What is your stress level (0-100)?'].describe())

data_preclean['What is your stress level (0-100)?'].to_csv('data.csv', index=False)