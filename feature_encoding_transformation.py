import pandas as pd
import seaborn as sns
import skillsnetwork
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv'
#await skillsnetwork.download_dataset(URL)
df = pd.read_csv('Ames_Housing_Data.tsv', sep='\t')
df.info()
df = df.loc[df['Gr Liv Area'] <= 4000,:]
print("Number of rows in the data:", df.shape[0])
print("Number of columns in the data:", df.shape[1])
data = df.copy()

#One hot encoding
one_hot_encode_cols = [column for column in df.columns if df[column].dtypes == object]
df[one_hot_encode_cols].head().T
df = pd.get_dummies(df, columns = one_hot_encode_cols, drop_first= True)
df.describe().T

#Log transforming skew variables
mask = data.dtypes ==float
float_cols = data.columns[mask]
skew_limit = 0.75
skew_vals = data[float_cols].skew()
skew_cols = (skew_vals.sort_values(ascending = True).to_frame().rename(columns = {0:'Skew'}).query('abs(Skew) > {}'.format(skew_limit)))


field = "BsmtFin SF 1"

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
df[field].hist(ax=ax_before)
df[field].apply(np.log1p).hist(ax=ax_after)
ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field));

#transform all skew col except 'SalePrice':
for col in skew_cols.index.values:
    if col == "SalePrice":
        continue
    df[col] = df[col].apply(np.log1p)

# There are a *lot* of variables. Let's go back to our saved original data and look at how many values are missing for each variable.
df = data
data.isnull().sum().sort_values()
smaller_df= df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond',
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area',
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces',
                      'Garage Cars','SalePrice']]
smaller_df.describe().T
smaller_df.info()
smaller_df = smaller_df.fillna(0)
smaller_df.info()
sns.pairplot(smaller_df, plot_kws = dict(alpha = .1, edgecolor = 'none'))

X = smaller_df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond',
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area',
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces',
                      'Garage Cars']]
y = smaller_df['SalePrice']

X2 = X.copy()
X2['OQ2'] = X2['Overall Qual']**2
X2['GLA2'] = X2['Gr Liv Area']**2
X3 = X2.copy()
X3['OQ_x_YB'] = X3['Overall Qual'] * X3['Year Built']
X3['OQ_/_LA'] = X3['Overall Qual'] / X3['Lot Area']
df['House Style'].value_counts()
pd.get_dummies(df['House Style'], drop_first=True).head()

nbh_counts = df.Neighborhood.value_counts()
nbh_counts
other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
X4 = X3.copy()
X4['Neighborhood'] = df['Neighborhood'].replace(other_nbhs, 'Other')

#Polynomial Features in Scikit-Learn
pf = PolynomialFeatures(degree = 2)
features = ['Lot Area', 'Overall Qual']
pf.fit(df[features])
pf.get_feature_names_out()
feat_array = pf.transform(df[features])
pd.DataFrame(feat_array, columns = pf.get_feature_names_out(input_features = features))
