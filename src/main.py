from functions.data_preprocessing import *
from functions.models import *
from functions.data_visualisation import *
import os

PRINT_INFO = False

def prepareAllCSVFiles():
  for dirname, _, filenames in os.walk('src/data/Original csvs/'):
      for filename in filenames:
          filepath = os.path.join(dirname, filename)
          new_df, df_names = preProcessColumns(filepath)

          new_df.to_csv('src/data/unprocessed_csvs/' + filename.split('.')[0] + '_unprocessed.csv')

def prepareData(filepath, y_column='IsDeadWithin4Months', drop_duplicates=False, test_size=0.3, random_state=0):
  df = readCSV(filepath)
  x, y = preProcessCSV(df, drop_duplicates=drop_duplicates, y_column=y_column)
  X_train, X_test, y_train, y_test = splitTrainingSet(x, y, test_size = test_size, random_state = random_state)

  return X_train, X_test, y_train, y_test

def plotData(model, X_test, y_test, filename):
  plotConfusionMatrix(model, X_test, y_test, 'M47_unprocessed.csv')
  calculateClassProbabilities(model, X_test)
  plotROCCurve(model, X_test, y_test)
  averagePrecisionScore(model, X_test, y_test)  

if __name__ == "__main__":

  dir_path = 'src/data/unprocessed_csvs'
  file_name = 'M47_unprocessed.csv'
  file_path = os.path.join(dir_path, file_name)
  classifier = "RandomForest"
  y_column = 'IsDeadWithin4Months'
  drop_duplicates = False
  test_size = 0.3
  random_state = 0

  ### Preparing all the CSV files. This only has to be done once ###
  # prepareAllCSVFiles()

  ### Preparing the data for the model ###
  X_train, X_test, y_train, y_test = prepareData(file_path, 
                                                 y_column, 
                                                 drop_duplicates, 
                                                 test_size, 
                                                 random_state)

  ### Can either optimise the hyperparameters or train the model, or run the the RF depth test  ###
  # optimiseHyperParameters(x, y, classifier)
  model = trainModel(X_train, X_test, y_train, y_test, classifier)
  # testRFDepth(X_train, X_test, y_train, y_test)

  ### Plotting the results ###
  plotData(model, X_test, y_test, file_name)

  ### Can also plot the decision trees if using RF ###
  # randomForestGraphs(model, X_train)

  ### Can also create the t-Tests for each column in the CSV file. Again this only has to be done once ###
  # df = readCSV(file_path)
  # x, y = preProcessCSV(df, drop_duplicates, y_column)
  # p_values = create_ttest(x, y)
 

  ### For testing all the CSV files provided by the NHS, not just the M47 dataset ###
  # for dirname, _, filenames in os.walk(dir_path):
  #     for filename in filenames:
  #         try:
  #             filepath = os.path.join(dirname, filename)
  #             X_train, X_test, y_train, y_test = prepareData(filepath, 
  #                                                             y_column, 
  #                                                             drop_duplicates, 
  #                                                             test_size, 
  #                                                             random_state)
  #             model = trainModel(X_train, X_test, y_train, y_test, classifier)
  #             plotConfusionMatrix(model, X_test, y_test, filename)
  #         except:
  #             continue
