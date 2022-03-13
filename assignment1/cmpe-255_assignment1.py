import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv(r"Levels_Fyi_Salary_Data.csv")
df.head()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df['year'] = df['timestamp'].dt.year
df.drop(["timestamp"], axis = 1, inplace=True)

df = df.drop_duplicates()

df = df.drop(columns = ['Race', 'Education', 'cityid', 'otherdetails', 'Some_College', 'Masters_Degree', 'Bachelors_Degree', 
                            'Doctorate_Degree', 'Race_Asian', 'Race_White', 'Race_Two_Or_More', 'level', 'tag', 
                            'Race_Black', 'Race_Hispanic', 'Highschool', 'rowNumber', 'dmaid', 'gender'])

# Since 'company' is an important feature for us let's replace the null values with "NA".

df['company'] = df['company'].fillna("NA")

## LabelEncoder and scaling

le_company = LabelEncoder()
df['company'] = le_company.fit_transform(df['company'])

le_location = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])

le_title = LabelEncoder()
df['title'] = le_title.fit_transform(df['title'])

## Modeling

X, y = df.iloc[:, :-1], df.iloc[:, -1]

p = 0.7

X_train = X.sample(frac = p, random_state = 42)
X_test = X.drop(X_train.index)
y_train = y.sample(frac = p, random_state = 42)
y_test = y.drop(y_train.index)

model = Sequential()
model.add(Dense(64, input_dim = len(X_train.iloc[0]), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics = 'mean_squared_error')

history = model.fit(X_train, y_train, epochs = 20)

y_pred = model.predict(X_test)
y_pred_flat = y_pred.flatten()

## Q1: How much you would get if I join for a position based on number of experiences and location?

position = 'Data Scientist'
experience = 4
location = 'Seattle, WA'

def predict_q1(position, experience, location):
  years_at_company, bonus = 2, 10000
  base_salary, stockgrantvalue = 150000, 20000
  year = 2023
  company = 'Amazon'
  
  tempComp = le_company.transform([company])[0]
  pos = le_title.transform([position])[0]
  loc = le_location.transform([location])[0]

  data = {'year': [year],'company': [tempComp],'position': [pos],'location': [loc],'experience': [experience], "years_at_company": [years_at_company], 
            'base_salary': [base_salary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
  df = pd.DataFrame(data)
  y_test_pred_sal = model.predict(df).flatten()
  print('Predicted Salary: ', y_test_pred_sal[0])

predict_q1(position, experience, location)

## Q2: How much you would get if you accept an offer for a position from X company based on number of experiences and location?

position = 'Data Scientist'
company = 'Oracle'
experience = 4
location = 'Seattle, WA'

def predict_q2(company, position, experience, location):
  years_at_company, bonus = 3, 10000
  base_salary, stockgrantvalue = 165000, 15000  
  year = 2022
  
  tempComp = le_company.transform([company])[0]
  pos = le_title.transform([position])[0]
  loc = le_location.transform([location])[0]

  data = {'year':[year],'company':[tempComp],'position':[pos],'location':[loc],'experience':[experience], "years_at_company": [years_at_company], 
            'base_salary': [base_salary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
  df = pd.DataFrame(data)
  y_test_pred_sal = model.predict(df).flatten()
  print('Predicted Salary: ', y_test_pred_sal[0])

predict_q2(company, position, experience, location)

## Q3: How much you will be getting for a position after Y years joining to X company?

position = 'Data Scientist'
company = 'Oracle'
years_at_company = 4

def predict_q3(years_at_company, position, company):
  experience, location = 3, 'Seattle, WA'
  bonus, year = 20000, 2023
  base_salary, stockgrantvalue = 175000, 10000

  tempComp = le_company.transform([company])[0]
  pos = le_title.transform([position])[0]
  loc = le_location.transform([location])[0]

  
  data = {'year':[year],'company':[tempComp],'position':[pos],'location':[loc],'experience':[experience], "years_at_company": [years_at_company], 
            'base_salary': [base_salary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
  df = pd.DataFrame(data)
  y_test_pred_sal = model.predict(df).flatten()
  print('Predicted Salary: ', y_test_pred_sal[0])

predict_q3(years_at_company, position, company)