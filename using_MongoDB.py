from mongoengine import *
import pandas as pd
import datetime

connect('tumblelog')


class ConHeo(Document):
    field1 = StringField()
    field2 = IntField()


conheo_con = ConHeo(field1='heo')
conheo_con.save()

all_conheo = ConHeo.objects(field1=10)
for conheo in all_conheo:
    print(conheo.field1)

#import XAUUSD price (H1 timeframe) CSV file into MongoDB
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('xauusd_H1.csv', delimiter= ',')
df.info()
df['Date'] = pd.to_datetime(df['Date'])

connect('tumblelog')

class XAUUSD_H1(Document):
    No = IntField()
    Date = DateTimeField(primary_key=True)
    timeframe = StringField()
    Open = FloatField()
    Close = FloatField()
    Low = FloatField()
    High = FloatField()
    Volume = IntField()



#for i in range(len(df)):
#    price = XAUUSD_H1(No = df.iloc[i,0], Date = df['Date'][i], Open = df['Open'][i],
#                      Close = df['Close'][i], Low = df['Low'][i], High = df['High'][i],
#                      Volume = df['Volume'][i])
#    price.save()


df1 = pd.read_csv('bank_credit_scoring.csv', delimiter= ',')
df1.info()

fields_dict = {
    'Unnamed: 0': IntField(),
    'ID': IntField(),
    'Customer_ID': IntField(),
    'Month': IntField(),
    'Name': StringField(),
    'Age': FloatField(),
    'SSN': FloatField(),
    'Occupation': StringField(),
    'Annual_Income': FloatField(),
    'Monthly_Inhand_Salary': FloatField(),
    'Num_Bank_Accounts': FloatField(),
    'Num_Credit_Card': FloatField(),
    'Interest_Rate': FloatField(),
    'Num_of_Loan': FloatField(),
    'Type_of_Loan': StringField(),
    'Delay_from_due_date': FloatField(),
    'Num_of_Delayed_Payment': FloatField(),
    'Changed_Credit_Limit': FloatField(),
    'Num_Credit_Inquiries': FloatField(),
    'Credit_Mix': StringField(),
    'Outstanding_Debt': FloatField(),
    'Credit_Utilization_Ratio': FloatField(),
    'Credit_History_Age': FloatField(),
    'Payment_of_Min_Amount': StringField(),
    'Total_EMI_per_month': FloatField(),
    'Amount_invested_monthly': FloatField(),
    'Payment_Behaviour': StringField(),
    'Monthly_Balance': FloatField(),
    'Credit_Score': StringField(),
}

meta_dict = {
    'collection': 'bank_credit_scoring',  # Collection name in MongoDB
    'indexes': [
        {'fields': ['ID'], 'unique': True},  # Example index
        {'fields': ['Name', 'Customer_ID']}                  # Example compound index
    ]
}
bank_credit_scoring = type('bank_credit_scoring', (Document,), {**fields_dict, 'meta': meta_dict})
for idx, row in df1.iterrows():
    document = bank_credit_scoring()
    for field in fields_dict.keys():
        setattr(document, field, row[field])
    document.save
print("All rows have been saved to the database.")



