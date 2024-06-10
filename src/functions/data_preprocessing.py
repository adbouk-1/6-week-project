from datetime import datetime
import pandas as pd

PRINT_INFO = False


def convert_to_int(df_column):
    """
    Convert a column of a DataFrame to integer values using pandas.factorize.
    This was done before oneHotEncoding was used to convert the columns to integers. 
    This was not used in the final analysis

    Parameters:
    df_column (pandas.Series): The column to be converted.

    Returns:
    new_col (numpy.ndarray): The converted column as an array of integers.
    uniques (pandas.Index): The unique values in the original column.

    """

    new_col, uniques = pd.factorize(df_column)

    return new_col, uniques
def preProcessColumns(csv_file, convert_to_int=False):
    """
    Preprocesses the columns of a CSV file, by calculating any additional columns needed for the model and converting string columns to integers.

    Args:
        csv_file (str): The path to the CSV file.
        convert_to_int (bool, optional): Indicates whether to convert string columns to integers. Defaults to False.

    Returns:
        tuple: A tuple containing two DataFrames:
            - df: The preprocessed DataFrame.
            - df_names: A DataFrame containing the unique values for each converted column.

    """

    df = pd.read_csv(csv_file)
    df_names = pd.DataFrame()

    # Create new columns for the target variables
    df['IsReadmitted'] = df['EmergencyReadmissionDateTime'].apply(lambda x: 0 if pd.isnull(x) else 1)

    df['MonthsAwayFromDeath'] = df['BKProviderSpellNumber']
    for i in range(len(df['IsDeceased'])):
        if df.loc[i, "IsDeceased"] == 'Yes':
            df.loc[i, "MonthsAwayFromDeath"] = (datetime.strptime(df.loc[i, "DateOfDeath"], '%d/%m/%Y').year - datetime.strptime(df.loc[i, "WardEndDateTime"], '%d/%m/%Y %H:%M').year) * 12 + (datetime.strptime(df.loc[i, "DateOfDeath"], '%d/%m/%Y').month - datetime.strptime(df.loc[i, "WardEndDateTime"], '%d/%m/%Y %H:%M').month)
        else:
            df.loc[i, "MonthsAwayFromDeath"] = 5

    df['IsDeadWithin4Months'] = df['MonthsAwayFromDeath'].apply(lambda x: 1 if x<5 else 0)

    if convert_to_int:
        # Change each column to integer if it is a string
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column], unique_values = convert_to_int(df[column])
                # If the new column has more unique values than the original DataFrame, reindex the DataFrame to make sure it can be concatenated
                if len(unique_values) > len(df_names):
                    new_index = range(len(unique_values))  # New index based on the longer Series
                    df_names = df_names.reindex(new_index)
                else:
                    unique_values = unique_values.to_series()
                df_names[column] = unique_values
    else:
        df_names = None

    return df, df_names


def readCSV(filepath):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The DataFrame containing the data from the CSV file.
    """

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
    """
    Preprocesses a CSV DataFrame by performing various data cleaning and transformation steps.

    Args:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
        drop_duplicates (bool, optional): Whether to drop duplicate rows based on the 'PatientPseudoNo' column. Defaults to False.
        y_column (str, optional): The target column to be used as the y variable. Can be either 'IsDeadWithin4Months' or 'IsReadmitted'. Defaults to 'IsDeadWithin4Months'.

    Returns:
        tuple: A tuple containing the preprocessed X DataFrame and the y Series.
    """

    #There may be some corrupt data where the value is '#NUM!' instead of 'NO' or 2 instead of 0. These need to be replaced.
    if df['IsDeadWithin4Months'].dtype!='O':
        df['IsDeadWithin4Months'] = df['IsDeadWithin4Months'].replace(2, 0)
    else:
        df['IsDeadWithin4Months'] = df['IsDeadWithin4Months'].replace('#NUM!', "NO")

    #Drop any duplicate patient rows from the dataset
    if drop_duplicates:
        df = df.drop_duplicates(subset=['PatientPseudoNo'])

    if PRINT_INFO:
        print(df['IsDeadWithin4Months'].value_counts())
    
    #This is a list of all the columns that were removed from the model for various reasons. These are explained in the report.
    x = df.drop(['IsDeceased', 'EmergencyReadmissionDateTime', 'DateOfDeath', 'UniqueEpisodeID', 'BKProviderSpellNumber',
             'EpisodeStartDateTime', 'EpisodeEndDateTime', 'Ward', 'EthnicOrigin', 'SourceSystem', 'TreatmentFunction.1', 
             'AdmissionDate', 'EpisodeDiagnosisCodeList', 'WardStartDateTime', 'WardEndDateTime', 'EpisodeProcedureCodeList', 'PatientPseudoNo', 
             'TreatmentFunctionCode.1', 'MonthsAwayFromDeath', 'DischargeMethod', 'DischargeDestination', 'TreatmentFunction', 'TreatmentFunctionCombined', 'Unnamed: 0',
             'SpecialtyKey', 'IPRDepartment', 'EthnicOriginGroup'], axis=1)
    
    #This is the target variable that we are trying to predict. It is either 'IsDeadWithin4Months' or 'IsReadmitted'
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
