import numpy as np
import pandas as pd
import datetime
import json
import plotly.express as px
import requests

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

gasoline = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/18100001.csv")
gasoline.head(10)
gasoline.info()
gasoline.isnull().sum()

#filter data needed
data = gasoline[['REF_DATE', 'GEO', 'Type of fuel', 'VALUE']].rename(columns = {'REF_DATE':'DATE', 'Type of fuel':'TYPE'})
data.describe(include = 'all')
data.isnull().sum()
#splitting City and Province
data[['City','Province']] = data['GEO'].str.split(',', n = 1 , expand = True)
data.head(5)

#Changing datetime format
data['DATE'] = pd.to_datetime(data['DATE'], format = '%b-%y')
data['Month'] = data['DATE'].dt.month_name().str.slice(stop = 3)
data['YEAR'] = data['DATE'].dt.year
data.describe(include = 'all')

data.GEO.unique().tolist()
data.TYPE.value_counts()

#data filter
mult_loc = data[(data['GEO'] == "Toronto, Ontario") | (data['GEO'] == "Edmonton, Alberta")]
mult_loc

cities = ['Calgary', 'Toronto', 'Edmonton']
CTE = data[data['City'].isin(cities)]
CTE.head(5)

geo = data.groupby('GEO')
geo.ngroups

group_year = data.groupby('YEAR')['VALUE'].mean()
group_year

#groupby and find parameter
exercise3b = data.groupby(['YEAR', 'City'])['VALUE'].median()
exercise3a = data.groupby(['YEAR', 'City'])['VALUE'].max()

#plot prices of gasoline in all cities during 1979 - 2021 use plotly.express
price_bycity = data.groupby(['YEAR','GEO'])['VALUE'].mean().reset_index(name = 'value').round(2)
fig = px.line(price_bycity, x = 'YEAR', y = 'value', color = 'GEO',
              color_discrete_sequence = px.colors.qualitative.Light24)
fig.update_traces(mode = 'markers + lines')
fig.update_layout(title = "Gasoline Price trend per City",
                  xaxis_title = 'YEAR',
                  yaxis_title = 'Annual Average Price, Cents per Litre')
fig.show()
#plot Toronto Price for year 2021
group_month = data[data['YEAR'] == 2021 & data['City'] == 'Toronto'].groupby(['Month'])['VALUE'].mean().reset_index().sort_values(by = 'VALUE')
fig = px.line(group_month, x = 'Month', y = 'VALUE')
fig.update_layout(title = 'Toronto Average Monthly Gasoline Price in 2021',
                    xaxis_title = 'Month',
                    yaxis_title = 'Monthly Price, Cents per Litre')
fig.show()

#Annual average gasoline price, per year, per gasoline type
group_type = data.groupby(['YEAR','TYPE'])['VALUE'].mean().reset_index().round(2)
fig = px.line(group_type, x = 'YEAR', y = 'VALUE', color = 'TYPE', color_discrete_sequence = px.colors.qualitative.Light24)
fig.update_traces(mode = 'markers + lines')
fig.update_layout(title = 'Gasoline price by TYPE through YEAR', xaxis_title = 'YEAR', yaxis_title = 'Annual average price')

#Plot average gasoline price by City for every year
bycity = data.groupby(['YEAR', 'City'])['VALUE'].mean().reset_index(name ='Value').round(2)
fig = px.bar(bycity, x = 'City', y = 'Value', color = 'City', animation_frame = 'YEAR')
fig.update_layout(title = 'Average Gasoline Price by City every year', xaxis_title = 'Year', yaxis_title = 'Average Price of Gasoline')
fig.show()

#plot a map
