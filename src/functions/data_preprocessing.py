import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import datetime

PRINT_INFO = False

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

    for dirname, _, filenames in os.walk('src/data/Original csvs/'):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            print(filepath)
            new_df, df_names = preprocess_csv(filepath)

            new_df.to_csv('src/data/unprocessed_csvs/' + filename.split('.')[0] + '_unprocessed.csv')

def pickFileName(file_number):
    for dirname, _, filenames in os.walk('src/data'):
        for filename in filenames:
            if PRINT_INFO:
                print(os.path.join(dirname, filename))


    filepath = os.path.join(dirname, filenames[file_number])
    return filepath

def readCSV(filepath):
    df = pd.read_csv(filepath)

    if PRINT_INFO:
        print(df.shape)
        print(df.info())

        categorical = [var for var in df.columns if df[var].dtype=='O']

        print('There are {} categorical variables\n'.format(len(categorical)))

        print('The categorical variables are :\n\n', categorical)
        
        for var in categorical: 
            print(var, ' contains ', len(df[var].unique()), ' labels')

        numerical = [var for var in df.columns if df[var].dtype!='O']

        print('There are {} numerical variables\n'.format(len(numerical)))
        print('The numerical variables are :', numerical)



    return df


def preProcessCSV(df: pd.DataFrame, drop_duplicates=False, y_column='IsDeadWithin4Months'):

    print(df['IsDeadWithin4Months'].dtype!='O')
    if df['IsDeadWithin4Months'].dtype!='O':
        df['IsDeadWithin4Months'] = df['IsDeadWithin4Months'].replace(2, 0)
    else:
        df['IsDeadWithin4Months'] = df['IsDeadWithin4Months'].replace('#NUM!', "NO")

    if drop_duplicates:
        df = df.drop_duplicates(subset=['PatientPseudoNo'])

    if PRINT_INFO:
        print(df['IsDeadWithin4Months'].value_counts())
    
    x = df.drop(['IsDeceased', 'EmergencyReadmissionDateTime', 'DateOfDeath', 'UniqueEpisodeID', 'BKProviderSpellNumber',
             'EpisodeStartDateTime', 'EpisodeEndDateTime', 'Ward', 'EthnicOrigin', 'SourceSystem', 'TreatmentFunction.1', 
             'AdmissionDate', 'EpisodeDiagnosisCodeList', 'WardStartDateTime', 'WardEndDateTime', 'EpisodeProcedureCodeList', 'PatientPseudoNo', 
             'TreatmentFunctionCode.1', 'MonthsAwayFromDeath', 'DischargeMethod', 'DischargeDestination', 'TreatmentFunction', 'TreatmentFunctionCombined', 'Unnamed: 0',
             'SpecialtyKey', 'IPRDepartment', 'EthnicOriginGroup'], axis=1)
    
    if y_column == 'IsDeadWithin4Months':
        x = x.drop(['IsDeadWithin4Months'], axis=1)
        y = df['IsDeadWithin4Months']
    elif y_column == 'IsReadmitted':
        x = x.drop(['IsReadmitted'], axis=1)
        y = df['IsReadmitted']

    if PRINT_INFO:
        print(x.shape)
        print(y.shape)
    
    return x, y
