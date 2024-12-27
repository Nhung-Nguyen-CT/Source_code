import  pandas as pd
import numpy as np

df = pd.read_csv()
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))

df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN','Month','Num_Bank_Accounts', 'Outstanding_Debt','Amount_invested_monthly', 'Monthly_Balance', 'Num_Credit_Inquiries', 'Monthly_Inhand_Salary'])
len_df = len(df)