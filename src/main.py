from functions.data_preprocessing import *
from functions.models import *
from functions.data_visualisation import *

PRINT_INFO = False

for dirname, _, filenames in os.walk('src/data/Original csvs/'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        print(filepath)
        new_df, df_names = preProcessColumns(filepath)

        new_df.to_csv('src/data/unprocessed_csvs/' + filename.split('.')[0] + '_unprocessed.csv')

# df = readCSV('src/data/unprocessed_csvs/M47_unprocessed.csv')
# x, y = preProcessCSV(df, drop_duplicates=False, y_column='IsDeadWithin4Months')
# graph_clustering(x)
# optimiseHyperParameters(x, y, classifier="RandomForest")
# X_train, X_test, y_train, y_test = splitTrainingSet(x, y, test_size = 0.3, random_state = 0, classifier="RandomForest")
# model = trainModel(X_train, X_test, y_train, y_test, classifier="RandomForest")
# test(X_train, X_test, y_train, y_test)
# plotConfusionMatrix(model, X_test, y_test, 'M47_unprocessed.csv')
# calculateClassProbabilities(model, X_test)
# plotROCCurve(model, X_test, y_test)
# averagePrecisionScore(model, X_test, y_test)  
for dirname, _, filenames in os.walk('src/data/unprocessed_csvs/'):
    for filename in filenames:
        try:
            filepath = os.path.join(dirname, filename)
            print(filepath)
            df = readCSV(filepath)
            x, y = preProcessCSV(df, drop_duplicates=False, y_column='IsDeadWithin4Months')
            # X_train, X_test, y_train, y_test = splitTrainingSet(x, y, test_size = 0.3, random_state = 0, classifier="RandomForest")
            # model = trainModel(X_train, X_test, y_train, y_test, classifier="RandomForest")
            # plotConfusionMatrix(model, X_test, y_test, filename)
        except:
            continue
# randomForest(model)

# df = readCSV('src/data/preprocessed_csvs/M47_preprocessed.csv')
# x, y = preProcessCSV(df, drop_duplicates=False, y_column='IsDeadWithin4Months')
# p_values = create_ttest(x, y)