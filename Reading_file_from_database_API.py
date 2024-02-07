import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd
from IPython.display import display

path = 'data/classic_rock.db'
con = sq3.Connection(path)

query = '''
SELECT *
FROM rock_songs;
'''
observations = pds.read_sql(query, con)
observations.head()

query = '''
SELECT Artist, Release_Year, Count(*) as num_songs, AVG(PlayCount) as avg_plays
FROM rock_songs
GROUP BY Artist, Release_Year
ORDER BY num_songs DESC
'''
observations = pds.read_sql(query, con)
observations.head()

#create churn table to easily observe the table
observations_generator = pds.read_sql(query,
                                      con,
                                      coerce_float = True,
                                      parse_dates = ['Release_Year'],
                                      chunksize = 5)
for index, observations in enumerate(observations_generator):
    if index < 5:
        print(f'Observations index: {index}'.format(index))
        display(observations)

