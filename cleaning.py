import pandas as pd
import numpy as np
import re

dataset = pd.read_csv('ODI-2023.csv', header=0, index_col=0, delimiter=';')
print(dataset.columns)


def standardize_program_name(name):
    name = name.lower().strip()
    if 'artificial intelligence' in name or 'masters - ai' in name or 'ai master' in name or 'master ai' in name:
        return 'Artificial Intelligence'
    return name

def convert_yes_no_to_bool(value):
    if value.lower() == 'yes':
        return True
    if value.lower() == 'no':
        return False
    return value

def convert_numeric_to_bool(value):
    if value == 1:
        return True
    if value == 0:
        return False
    return value

def standardize_gender(gender):
    gender = gender.lower().strip()
    if gender == 'male' or gender == 'm':
        return 'Male'
    elif gender == 'female' or gender == 'f':
        return 'Female'
    else:
        return 'Other'

def standardize_chatgpt_help(help_response):
    help_response = help_response.lower().strip()
    if help_response in ['yes', 'y', 'true']:
        return 1
    elif help_response in ['no', 'n', 'false']:
        return 0
    else:
        return np.nan

def clean_student_count(estimate):
    try:
        return int(estimate)
    except ValueError:
        return np.nan

def standardize_stand_up(stand_up):
    stand_up = stand_up.lower().strip()
    if stand_up in ['yes', 'y', 'true']:
        return 1
    elif stand_up in ['no', 'n', 'false']:
        return 0
    else:
        return np.nan

def clean_stress_level(stress_level):
    try:
        stress_level = float(stress_level)
        if 0 <= stress_level <= 100:
            return stress_level
        else:
            return np.nan
    except ValueError:
        return np.nan

def clean_sports_hours(hours):
    try:
        return int(hours)
    except ValueError:
        return np.nan

def clean_random_number(random_number):
    try:
        return float(random_number)
    except ValueError:
        return np.nan

def parse_bed_time(time_str):
    try:
        time = pd.to_datetime(time_str, format='%H:%M').time()
        return time
    except ValueError:
        return np.nan

def preprocess_text(text):
    if text != pd.notna:
        pass
    elif text == '\u2615':
        print(text)
        text = 'coffee'
    else:
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    return text






dataset['What programme are you in?'] = dataset['What programme are you in?'].apply(standardize_program_name)
dataset['Have you taken a course on machine learning?'] = dataset['Have you taken a course on machine learning?'].apply(convert_yes_no_to_bool)
dataset['Have you taken a course on information retrieval?'] = dataset['Have you taken a course on information retrieval?'].apply(convert_numeric_to_bool)

# Reuse the convert_yes_no_to_bool function
dataset['Have you taken a course on statistics?'] = dataset['Have you taken a course on statistics?'].apply(convert_yes_no_to_bool)


# Reuse the convert_numeric_to_bool function
dataset['Have you taken a course on databases?'] = dataset['Have you taken a course on databases?'].apply(convert_numeric_to_bool)


dataset['What is your gender?'] = dataset['What is your gender?'].apply(standardize_gender)

# Reuse the datetime conversion from earlier for the 'Tijdstempel' column
dataset['When is your birthday (date)?'] = pd.to_datetime(dataset['When is your birthday (date)?'], errors='coerce')

dataset['How many students do you estimate there are in the room?'] = dataset['How many students do you estimate there are in the room?'].apply(clean_student_count)


dataset['I have used ChatGPT to help me with some of my study assignments '] = dataset['I have used ChatGPT to help me with some of my study assignments '].apply(standardize_chatgpt_help)

dataset['When is your birthday (date)?'] = pd.to_datetime(dataset['When is your birthday (date)?'], errors='coerce')

dataset['Did you stand up to come to your previous answer    ?'] = dataset['Did you stand up to come to your previous answer    ?'].apply(standardize_stand_up)


dataset['What is your stress level (0-100)?'] = dataset['What is your stress level (0-100)?'].apply(clean_stress_level)


dataset['How many hours per week do you do sports (in whole hours)? '] = dataset['How many hours per week do you do sports (in whole hours)? '].apply(clean_sports_hours)

dataset['Give a random number'] = dataset['Give a random number'].apply(clean_random_number)

dataset['Time you went to bed Yesterday'] = dataset['Time you went to bed Yesterday'].apply(parse_bed_time)
dataset['What makes a good day for you?'] = dataset['What makes a good day for you (1)?'] + ' ' + dataset['What makes a good day for you (2)?']
dataset['What makes a good day for you?'] = dataset['What makes a good day for you?'].apply(preprocess_text)
dataset.drop(['What makes a good day for you (1)?', 'What makes a good day for you (2)?'], axis=1, inplace=True)





print(dataset.head(3))
with open("asd.xlsx", "w", encoding='utf-8') as out:
    out.write(dataset.to_excel('output.xlsx', engine='openpyxl'))
