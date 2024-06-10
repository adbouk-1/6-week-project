import pandas as pd
from datetime import datetime
import os

def check_is_boolean(df_column, yes_value, no_value):
    new_col = df_column.replace({no_value: 0, yes_value: 1})
    return new_col

def convert_to_int(df_column):
    new_col, uniques = pd.factorize(df_column)

    return new_col, uniques
def preprocess_csv(csv_file):
    df = pd.read_csv(csv_file)
    df_names = pd.DataFrame()

    #Add emergency readmission boolean
    df['IsReadmitted'] = df['EmergencyReadmissionDateTime'].apply(lambda x: 0 if pd.isnull(x) else 1)

    df['MonthsAwayFromDeath'] = df['BKProviderSpellNumber']
    for i in range(len(df['IsDeceased'])):
        if df.loc[i, "IsDeceased"] == 'Yes':
            df.loc[i, "MonthsAwayFromDeath"] = (datetime.strptime(df.loc[i, "DateOfDeath"], '%d/%m/%Y').year - datetime.strptime(df.loc[i, "WardEndDateTime"], '%d/%m/%Y %H:%M').year) * 12 + (datetime.strptime(df.loc[i, "DateOfDeath"], '%d/%m/%Y').month - datetime.strptime(df.loc[i, "WardEndDateTime"], '%d/%m/%Y %H:%M').month)
        else:
            df.loc[i, "MonthsAwayFromDeath"] = 5

    df['IsDeadWithin4Months'] = df['MonthsAwayFromDeath'].apply(lambda x: 1 if x<5 else 0)

    #change each column to integer if it is a string
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column], unique_EthnicOriginCode = convert_to_int(df[column])
            if len(unique_EthnicOriginCode) > len(df_names):
                new_index = range(len(unique_EthnicOriginCode))  # New index based on the longer Series
                df_names = df_names.reindex(new_index)
            else:
                unique_EthnicOriginCode = unique_EthnicOriginCode.to_series()
            df_names[column] = unique_EthnicOriginCode

    return df, df_names


        # df_names.to_csv('src/data/row_names/' + filename.split('.')[0] + '_row_names.csv')