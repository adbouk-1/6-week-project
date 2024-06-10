import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

SHOW_PLOTS = False
SAVE_PLOTS = True

def process_and_plot_binary(csv_file, output_filename, x_column, y_column):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    grouped = df.groupby([x_column, y_column]).size().unstack(fill_value=0)
    
    # Plotting
    ax = grouped.plot(kind='bar', stacked=False)
    ax.set_title(f'Comparison of {x_column} based on {y_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel('Counts')
    plt.xticks(rotation=45)  # Rotate labels to prevent overlap
    plt.legend(title=y_column)
    
    if SHOW_PLOTS:
        plt.show()

    if SAVE_PLOTS:
        plt.savefig(output_filename, format='png')  # Adjust format as needed

def process_and_plot_box(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # # #Fix or remove outliers
    for col in df.columns: 
        try:
            plt.boxplot(df[col])
            output_filename = "src/plots/box/" + col + "_box.png"
            
            if SHOW_PLOTS:
                plt.show()

            if SAVE_PLOTS:
                plt.savefig(output_filename, format='png')  # Adjust format as needed
            plt.clf()
        except:
            print(f"Could not plot box for {col}")
            continue


def process_and_plot_scatter(csv_file, output_filename, x_column, y_column):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Remove duplicates based on the specified column
    # df = df.drop_duplicates(subset=[duplicate_column])
    # filtered_df = df[df['EmergencyReadmissionDateTime'].notnull()]
    # #Exploratory Data Analysis (EDA)
    # # sns.pairplot(df)
    # # sns.distplot(df[x_column])
    # # sns.countplot(df[x_column])


    # # # Plotting the scatter plot
    plt.scatter(df[x_column], df[y_column])
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

    plt.savefig(output_filename, format='png')  # Adjust format as needed

def process_and_plot_regression(csv_file, output_filename, x_column, y_column):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Remove duplicates based on the specified column
    # df = df.drop_duplicates(subset=[duplicate_column])
    # filtered_df = df[df['EmergencyReadmissionDateTime'].notnull()]
    # #Exploratory Data Analysis (EDA)
    # # sns.pairplot(df)
    # sns.distplot(df[x_column])
    # sns.countplot(df[x_column])

    # sns.regplot(x = x_column, y = y_column, data = df)
    # plt.show()
    # Creating the correlation matrix of the iris dataset
    df_numeric = df.select_dtypes(exclude=['object', 'string'])
    iris_corr_matrix = df_numeric.corr()
    print(iris_corr_matrix)

    # Create the heatmap using the `heatmap` function of Seaborn
    sns.heatmap(iris_corr_matrix, cmap='coolwarm', annot=True)
    plt.show()

    # plt.savefig(output_filename, format='png')  # Adjust format as needed


# Example usage
columns = ['UniqueEpisodeID', 'BKProviderSpellNumber',
       'EpisodeSequenceNo', 'EpisodeStartDateTime', 'EpisodeEndDateTime',
       'PrimaryProcedureKey', 'ProcedureGroupKey', 'EpisodeProcedureCodeList',
       'PatientPseudoNo', 'Ward', 'WardCode', 'WardStartDateTime',
       'WardEndDateTime', 'LOSMinutes_Episode', 'LOSMinutes_Spell',
       'TreatmentFunctionCode', 'TreatmentFunction',
       'EpisodeDiagnosisCodeList', 'PrimaryDiagnosis', 'IsDeceased',
       'DateOfDeath', 'AdmissionDate', 'EmergencyReadmissionDateTime',
       'AgeAtAdmission', 'Gender', 'EthnicOriginCode', 'EthnicOrigin',
       'EthnicOriginGroup', 'DischargeMethod', 'DischargeDestination',
       'AdmissionSource', 'AdmissionMethod', 'SpecialtyKey', 'SourceSystem',
       'LocalSpecialtyCode', 'LocalSpecialty', 'TreatmentFunctionCode.1',
       'TreatmentFunction.1', 'TreatmentFunctionCombined', 'IPRDepartment',
       'IsConsultantLedService', 'IsASISpecialty', 'SpecialtyOwner',
       'TrustDivision', 'EmergencyReadmission']

int_columns = ['PrimaryProcedureKey', 'ProcedureGroupKey', #'WardCode', 'PrimaryDiagnosis', 
       'TreatmentFunctionCode', 'Gender', 'EthnicOriginCode',
       'EthnicOriginGroup', 'DischargeMethod', 'DischargeDestination', 'AgeAtAdmission',
       'AdmissionSource', 'AdmissionMethod', 'SpecialtyKey', 'SourceSystem',
       'LocalSpecialtyCode', 'TreatmentFunctionCode.1',
       'IPRDepartment', 'IsConsultantLedService', 'IsASISpecialty', 'SpecialtyOwner',
       'TrustDivision']

out_columns = ['EmergencyReadmission', 'IsDeceased', 'LOSMinutes_Episode', 'LOSMinutes_Spell']

csv_file = 'src/data/M47_Preprocessed.csv'  # Replace 'your_data.csv' with your CSV file name
x_column = 'AgeAtAdmission'  # Replace with the column for the x-axis of the scatter plot
y_column = 'IsDeceased'  # Replace with the column for the y-axis of the scatter plot
output_filename = "src/plots/" + x_column + "-v-" + y_column + ".png"

# process_and_plot_binary(csv_file, output_filename, x_column, y_column)

# process_and_plot_box(csv_file)

# for col in int_columns:
#     x_column = col
#     y_column = 'EmergencyReadmission'
#     output_filename = "src/plots/bar_readmission/" + x_column + "-v-" + y_column + ".png"
#     process_and_plot_binary(csv_file, output_filename, x_column, y_column)

for col in int_columns:
    x_column = col
    y_column = 'LOSMinutes_Episode'
    output_filename = "src/plots/bar_readmission/" + x_column + "-v-" + y_column + ".png"
    process_and_plot_regression(csv_file, output_filename, x_column, y_column)


# for col in int_columns:
#     x_column = col
#     y_column = 'LOSMinutes_Episode'
#     output_filename = "src/plots/scatter/" + x_column + "-v-" + y_column + ".png"
#     process_and_plot_scatter(csv_file, output_filename, x_column, y_column)
