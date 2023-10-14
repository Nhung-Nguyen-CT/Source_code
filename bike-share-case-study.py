import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime

#set up data view
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


# read files from 24 excel files of 2021 - 2022 bike-share info
print('READ DATA: 24 FILES (MONTH) IN 2021')
data = {}
data_col_names = {}
year = 2021
for i in range(2):
    year = year + i
    for j in range(1,13) :
            if j<10 :
                month = f"0{j}"
            else:
                month = f"{j}"
            file_path = rf"./copy-data/{year}{month}-divvy-tripdata.csv"
            data[f'{year}_{j}'] = pd.read_csv(file_path)
            data_col_names[f'{year}_{j}'] = data[f'{year}_{j}'].columns.tolist()

print('Done read file')
print('')
# check compatibility table
print('CHECK COMPATIBILITY OF TABLES:')
col_names = np.sort(np.unique(np.concatenate(list(data_col_names.values()))))
print("all column names: ",col_names)
print("column names of 2021_1: ",data_col_names['2021_1'])
if len(col_names) == len(data_col_names['2021_1']) :
    print('Not difference between columns names')
else:
    print('Differences between columns names')
    print('')
# result: correct column names

#Concat dataframes
print('CONCAT TABLES')
bikeshare_concat_df = pd.concat(data)
print('10 last row: ',bikeshare_concat_df[-10:])
print('final data info: ', bikeshare_concat_df.info())
print('Done concat')
print('')

#change type to standard type
print('STANDARDIZE DATATYPE')
bikeshare_df = bikeshare_concat_df.convert_dtypes()
bikeshare_df["started_at"] = pd.to_datetime(bikeshare_df.started_at, format = '%Y-%m-%d %H:%M:%S')
bikeshare_df["ended_at"] = pd.to_datetime(bikeshare_df.ended_at, format = '%Y-%m-%d %H:%M:%S')
print('converted datatype info: ')
data_converted_info = bikeshare_df.info()
print('Done standardize datatype')
print('')



#Cleaning data
print('MORE CLEANING DATA: ')
# remove duplicate rows
bikeshare_df = bikeshare_df.drop_duplicates(keep='first')
# ride_id: len and duplicate
ride_id_len = bikeshare_df['ride_id'].str.len()
if len(pd.unique(ride_id_len)) == 1 :
    print('Length of ride_id is consistent ')
else:
    print('Length of ride_id is not consistent ')
if pd.unique(bikeshare_df['ride_id']) == len(bikeshare_df) :
    print('No duplicate ride_id')
else:
    print('Duplicated ride_id')

# check unique of rideable_type
print("Unique value of rideable_type: ",pd.unique(bikeshare_df['rideable_type']))

#check mapping of start station name and start station id, end station name and end station id
print('Start station name vs start station id:')
df1 = bikeshare_df.groupby(['start_station_name'])['start_station_id'].nunique()
repeat_name_IDs_start =  df1.iloc[np.where(df1 > 1)[0]].index
if len(repeat_name_IDs_start ) != 0 :
    repeat_name_IDs_start_df = bikeshare_df[bikeshare_df['start_station_name'].isin(repeat_name_IDs_start)]
    wrong_name_IDs_start = repeat_name_IDs_start_df.groupby(['start_station_name','start_station_id'])['start_station_id'].count()
    print('Mapping error: 1 name mapping with more than 1 ID')
    print(repeat_name_IDs_start)
    print(wrong_name_IDs_start)
df2 = bikeshare_df.groupby(['start_station_id'])['start_station_name'].nunique()
repeat_ID_names_start =  df2.iloc[np.where(df2 > 1)[0]].index
if len(repeat_ID_names_start) != 0 :
    repeat_ID_names_start_df = bikeshare_df[bikeshare_df['start_station_id'].isin(repeat_ID_names_start)]
    wrong_ID_names_start = repeat_ID_names_start_df.groupby(['start_station_id', 'start_station_name'])['start_station_name'].count()
    print('Mapping error: 1 ID mapping with more than 1 name: ')
    print(repeat_ID_names_start)
    print(wrong_ID_names_start)
print('End station name vs end station id:')
df1 = bikeshare_df.groupby(['end_station_name'])['end_station_id'].nunique()
repeat_name_IDs_end =  df1.iloc[np.where(df1 > 1)[0]].index
if len(repeat_name_IDs_end ) != 0 :
    repeat_name_IDs_end_df = bikeshare_df[bikeshare_df['end_station_name'].isin(repeat_name_IDs_end)]
    wrong_name_IDs_end = repeat_name_IDs_end_df.groupby(['end_station_name', 'end_station_id'])['end_station_id'].count()
    print('Mapping error: 1 name mapping with more than 1 ID')
    print(repeat_name_IDs_end)
    print(wrong_name_IDs_end)
df2 = bikeshare_df.groupby(['end_station_id'])['end_station_name'].nunique()
repeat_ID_names_end =  df2.iloc[np.where(df2 > 1)[0]].index
if len(repeat_ID_names_end) != 0 :
    repeat_ID_names_end_df = bikeshare_df[bikeshare_df['end_station_id'].isin(repeat_ID_names_end)]
    wrong_ID_names_end = repeat_ID_names_end_df.groupby(['end_station_id', 'end_station_name'])['end_station_name'].count()
    print('Mapping error: 1 ID mapping with more than 1 name')
    print(repeat_ID_names_end)
    print(wrong_ID_names_end)
#write to excel file to know how to replace
file_path = r"./copy-data/check_mapping_id_name_2years.xlsx"
with pd.ExcelWriter(file_path) as writer:
    wrong_name_IDs_start.to_excel(writer,sheet_name = "wrong_name_IDs")
    wrong_name_IDs_end.to_excel(writer,sheet_name = "wrong_name_IDs", startrow=0, startcol=10)
    wrong_ID_names_start.to_excel(writer,sheet_name = "wrong_ID_names")
    wrong_ID_names_end.to_excel(writer,sheet_name = "wrong_ID_names", startrow=0, startcol=10)
# replace ID and name with correct
file_path = rf"./copy-data/re_map_bikeshare.xlsx.csv"
name_IDs_df = pd.read_excel(file_path, sheet_name="re_map_name_ID")
ID_names_df = pd.read_excel(file_path, sheet_name="re_map_ID_name")
name_IDs_df = name_IDs_df.set_index('station_name')
name_IDs_dict = name_IDs_df.to_dict('index')
ID_names_df = ID_names_df.set_index('station_id')
ID_names_dict = ID_names_df.to_dict('index')
# replace correct ID names
#
bikeshare_df = bikeshare_df.replace({'start_station_id': )
bikeshare_df = bikeshare_df.replace({'start_station_name': replace_list, 'end_station_name': replace_list })
print('Done re-mapping names and IDs')
#Check member_casual column
print("Unique value of member_casual: ",pd.unique(bikeshare_df['member_casual']))

#missing value
print('Number of non NA: ',bikeshare_df.count())
print('Missing values column: start_station_name, start_station_id, end_station_name, end_station_id')
print('Delete rows that contain 4 NA values of 4 columns above')
bikeshare_df =  bikeshare_df.dropna(thresh=10)

#sort data by start time
bikeshare_df.sort_values(by=['started_at'], ascending=True)
print("Done sort data by start_at")
print("Done cleaning data")
print("")

#Data transform
#create 2 more column: length of a biking and weekday of started day
print('CREATE 2 MORE COLUMNS: LENGTH OF A BIKING AND WEEK OF STARTED DAY')
bikeshare_df["biking_length"] = bikeshare_df["ended_at"] - bikeshare_df["started_at"]
bikeshare_df['weekday'] = bikeshare_df['started_at'].dt.dayofweek
print('Done create columns')
print('')

print('Done')



