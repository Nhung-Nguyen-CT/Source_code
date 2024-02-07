import sqlite3
import numpy as np
import pandas as pd
import pandas.io.sql as pds

path = 'data/baseball.db'
con = sqlite3.Connection(path)
query = '''
SELECT *
FROM allstarfull
'''
allstarfull_observations = pd.read_sql(query, con)
allstarfull_observations.head(10)

all_tables = pd.read_sql('SELECT * FROM sqlite_master', con)
print(all_tables)

best_query = '''
SELECT playerID, sum(GP) as num_games_played, AVG(startingPos) as avg_starting_position
FROM allstarfull
GROUP BY playerID
ORDER BY num_games_played DESC, avg_starting_position ASC
LIMIT 3
'''
best = pd.read_sql(best_query, con)
print(best.head())
