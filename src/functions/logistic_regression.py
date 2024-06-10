import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.naive_bayes import GaussianNB
import os
from sklearn import svm
from sklearn.tree import export_graphviz
import graphviz
from scipy.stats import ttest_ind
import networkx as nx
import community.community_louvain as community_louvain
import seaborn as sns

PRINT_INFO = False

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

def plotCorrelationMatrix(df):
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, cmap='coolwarm', mask=mask, annot=True, fmt='.1g', xticklabels=corr.columns, yticklabels=corr.columns, cbar=False)
    plt.xticks(fontsize=8)  # Rotate labels by 45 degrees
    plt.show()

def splitTrainingSet(x, y, test_size = 0.3, random_state = 0):
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
     
        categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
        encoder = ce.OneHotEncoder(cols=categorical)

        X_train = encoder.fit_transform(X_train)
        X_test = encoder.transform(X_test)

        return X_train, X_test, y_train, y_test

def trainModel(X_train, X_test, y_train, y_test, classifier="LogisticRegression"):

    if classifier == "LogisticRegression":
        # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
        #                    intercept_scaling=1, l1_ratio=None, max_iter=100,
        #                    multi_class='warn', n_jobs=None, penalty='l2',
        #                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        #                    warm_start=False)
        #Readmission rate: 4.281332398719396
        #Dead within 4 months: 545.5594781168514    
        model = LogisticRegression(solver='liblinear', C=545.5594781168514, random_state=0, penalty='l1')

        model.fit(X_train, y_train)

        if PRINT_INFO:
            print(model.classes_)
            print(model.intercept_)
            print(model.coef_)

            model.predict_proba(X_test)

            model.predict(X_test)

            print(model.score(X_test, y_test))

            y_pred = model.predict(X_test)
            print(y_pred)
            print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

            y_pred_train = model.predict(X_train)
            print(y_pred_train)
            print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


            #Check null acuracy
            print(y_test.value_counts())
            print(np.unique(y_pred))
            null_accuracy = (y_test.value_counts().max()/(y_test.value_counts().max()+y_test.value_counts().min()))
            print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
    elif classifier == "NaiveBayes":

        model = GaussianNB(var_smoothing=8.111308307896872e-05)
        model.fit(X_train, y_train)

        if PRINT_INFO:
            #Predict the results
            y_pred = model.predict(X_test)
            print(y_pred)
            print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

            y_pred_train = model.predict(X_train)
            print(y_pred_train)
            print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


            #Check null acuracy
            print(y_test.value_counts())
            print(np.unique(y_pred))
            null_accuracy = (y_test.value_counts().max()/(y_test.value_counts().max()+y_test.value_counts().min()))
            print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
    elif classifier=="SVM":
        model = svm.SVC(kernel='linear', C=1, random_state=0, verbose=1)
        model.fit(X_train, y_train)

        if PRINT_INFO:
            print(model.classes_)
            print(model.intercept_)
            print(model.coef_)

            model.predict_proba(x)

            model.predict(x)

            print(model.score(X_test, y_test))
    elif classifier=="RandomForest":
        # model = RandomForestClassifier(max_depth=14, n_estimators=190)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        if PRINT_INFO:

            print(model.score(X_test, y_test))

            y_pred = model.predict(X_test)
            print(y_pred)
            print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

            y_pred_train = model.predict(X_train)
            print(y_pred_train)
            print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


            #Check null acuracy
            print(y_test.value_counts())
            print(np.unique(y_pred))
            null_accuracy = (y_test.value_counts().max()/(y_test.value_counts().max()+y_test.value_counts().min()))
            print('Null accuracy score: {0:0.4f}'. format(null_accuracy))



    return model

# Setup the logistic regression model
def optimiseHyperParameters(x, y, classifier="LogisticRegression"):

    X_train, X_test, y_train, y_test = splitTrainingSet(x, y, test_size = 0.3, random_state = 0)

    if classifier == "LogisticRegression":
        model = LogisticRegression()

        # Define the grid of hyperparameters to search
        param_grid = {
            'C': np.logspace(-100, 500, 50),  # Regularization strength
            'penalty': ['l1', 'l2'],      # Types of regularization
            'solver': ['liblinear']       # Solver capable of handling l1 penalty
        }

        # Setup the grid search with cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    elif classifier == "NaiveBayes":
        model = GaussianNB()

        # Define the grid of hyperparameters to search
        param_grid = {
            'var_smoothing': np.logspace(0,-9, num=100) # Variance smoothing parameter
        }

        # Setup the grid search with cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    elif classifier == "RandomForest":
        model = RandomForestClassifier()

        param_grid = {
            # 'n_estimators': randint(50,500), # Number of trees in the forest
            'max_depth': np.arange(65, 80, 1).tolist()
        }

        grid_search = RandomizedSearchCV(model, 
                                        param_distributions = param_grid, 
                                        n_iter=5, 
                                        cv=5)

# Fit the random search object to the data
    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", test_accuracy)



def plotConfusionMatrix(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    # plt.title('Confusion Matrix for ' + title.strip('_unprocessed.csv') + ' dataset trained on the M47 dataset')
    plt.title('Confusion Matrix for ' + title.strip('_unprocessed.csv') + ' dataset trained on the ' + title.strip('_unprocessed.csv') + ' dataset')
    plt.show()

    print(classification_report(y_test, model.predict(X_test)))

    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Classification error : {0:0.4f}'.format(classification_error))

    precision = TP / float(TP + FP)
    print('Precision : {0:0.4f}'.format(precision))

    recall = TP / float(TP + FN)
    print('Recall or Sensitivity : {0:0.4f}'.format(recall))

    false_positive_rate = FP / float(FP + TN)
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

    specificity = TN / (TN + FP)
    print('Specificity : {0:0.4f}'.format(specificity))

    return cm

def calculateClassProbabilities(model, X_test):
    # Predict the probability of each class
    y_pred1 = model.predict_proba(X_test)[:, 1]

    # plot histogram of predicted probabilities
    # adjust the font size 
    plt.rcParams['font.size'] = 12


    # plot histogram with 10 bins
    plt.hist(y_pred1, bins = 10)


    # set the title of predicted probabilities
    plt.title('Histogram of predicted probabilities of deaths within 4 months')


    # set the x-axis limit
    plt.xlim(0,1)


    # set the title
    plt.xlabel('Predicted probabilities of deaths within 4 months')
    plt.ylabel('Frequency')
    plt.show()

def plotROCCurve(model, X_test, y_test):
    # Predict the probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Import the roc_curve function
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(6,4))

    # Plot the ROC curve
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )

    # Set the limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.rcParams['font.size'] = 12
    # Set the labels and title
    plt.title('ROC curve for deaths within 4 months classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')

    # Add a grid
    plt.grid(True)

    # Add a diagonal line
    plt.plot([0, 1], [0, 1], 'k--')

    # Show the plot
    plt.show()

    # Import the roc_auc_score function
    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print('ROC AUC : {:.4f}'.format(roc_auc))

    Cross_validated_ROC_AUC = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc').mean()

    print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

    return roc_auc_score(y_test, y_pred_prob)

def kfoldCrossValidateModel(model, X_train, y_train):
    # Setup the cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

    # Print the cross-validation scores
    print('Cross-validation scores:{}'.format(cv_scores))
    # Print the average cross-validation score
    print('Average cross-validation score: {:.4f}'.format(cv_scores.mean()))

    return cv_scores

def averagePrecisionScore(model, X_test, y_test):
    # Import the precision_score function
    # Calculate the precision
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    average_precision = average_precision_score(y_test, y_pred_prob)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

    plt.plot(recall, precision, marker='.', label='Logistic')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.legend()

    plt.title(f'Precision Recall Curve. AUPRC: {average_precision}')

    plt.show()
    # print('Precision : {:.4f}'.format(precision))

def randomForest(model):
    for i in range(len(model.estimators_)):
        tree = model.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=X_train.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
        graph.render('./figures/Source_' + str(i) + '.gv', format='jpg',view=True)

def create_ttest(df, y):
    # Create a ttest



    categorical = [col for col in df.columns if df[col].dtypes == 'O']

    # Create a list of p-values
    p_values = []

    # Loop through the columns
    for col in df.columns:
        # Perform t-test
        x = df[col]

        print(col)

        if col in categorical:
            
            encoder = ce.OneHotEncoder(cols=col, use_cat_names=True)
            x = encoder.fit_transform(x)

        df_0 = x[y == 0]
        df_1 = x[y == 1]
        
        p_values = []
        counts = []
        original_values = []
        deaths = []

        if col in categorical:
            for cols in x.columns:
                ttest = ttest_ind(df_0[cols], df_1[cols])
                count = x[cols].sum()
                original_value = cols.split('_')[1]
                p_values.append(ttest.pvalue)
                counts.append(count)
                original_values.append(original_value)
                deaths.append(df_1[cols].sum())

            p_values = pd.DataFrame(p_values, index=x.columns, columns=['p_value'])
            p_values['count'] = counts
            p_values['original_value'] = original_values
            p_values['deaths'] = deaths
            # Sort the p-values
            p_values = p_values.sort_values(by='p_value')
            p_values.to_csv('src/data/p_values/' + col + '_ttest_results.csv')
        else:
            ttest = ttest_ind(df_0, df_1)
            p_values = pd.DataFrame([ttest.pvalue], columns=['p_value'])

            # Sort the p-values
            p_values = p_values.sort_values(by='p_value')
            p_values.to_csv('src/data/p_values/' + col + '_ttest_results.csv')


    return p_values

def test(X_train, X_test, y_train, y_test):
    random_state = 123

    output_tree = []
    output_rf = []

    # build initial forest
    initial_forest = RandomForestClassifier(
        max_features="sqrt", random_state=random_state
    )
    initial_forest_fit = initial_forest.fit(X_train, y_train)
    # get max depth
    max_depth_rf = max(
        [estimator.get_depth() for estimator in initial_forest_fit.estimators_]
    )
    # print(max_depth_rf)
    #construct depth grid
    list_depths_rf = np.arange(1, max_depth_rf, 1).tolist()
    depth_pct_rf = [100] + [(n / max_depth_rf) * 100 for n in list_depths_rf]

    # fit rf with smaller depth
    list_forests = [initial_forest_fit]
    print(max_depth_rf)
    for depth in list_depths_rf:
        tmp_forest = RandomForestClassifier(
            max_features="sqrt", max_depth=depth, random_state=random_state
        )
        list_forests.append(tmp_forest.fit(X_train, y_train))

    train_accuracy_rf = []
    test_accuracy_rf = []

    for forest in list_forests:
        # get accuracy for the forest
        yhat_train = forest.predict(X_train)
        yhat_test = forest.predict(X_test)
        train_accuracy_rf.append(accuracy_score(y_train, yhat_train))
        test_accuracy_rf.append(accuracy_score(y_test, yhat_test))

    df_rf = pd.DataFrame(
        list(
            zip(
                depth_pct_rf,
                train_accuracy_rf,
                test_accuracy_rf,
            )
        ),
        columns=["depth", "train_accuracy", "test_accuracy"],
    )
    output_rf.append(df_rf)

    df_rf.to_csv(path_or_buf="sim_shift_results_rf_check.csv", index=False)

def graph_clustering(df):
    # Create a graph
    G = nx.Graph()

    categorical = [col for col in df.columns if df[col].dtypes == 'O']
    encoder = ce.OneHotEncoder(cols=categorical)

    df = encoder.fit_transform(df)
    
    # Add nodes
    for col in df.columns:
        G.add_node(col)

    # Add edges based on correlation threshold
    threshold = 0.5  # Set a threshold for strong correlation
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            correlation = df[df.columns[i]].corr(df[df.columns[j]])
            if abs(correlation) > threshold:  # only consider strong correlations
                G.add_edge(df.columns[i], df.columns[j], weight=correlation)

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Add community info to the graph
    for node, comm_id in partition.items():
        G.nodes[node]['community'] = comm_id

    # Print the community of each node
    print("Node communities:")
    for node in G.nodes(data=True):
        print(node)

    color_map = [G.nodes[node]['community'] for node in G]
    pos = nx.spring_layout(G)  # positions for all nodes

    nx.draw(G, pos, node_color=color_map, with_labels=True, cmap=plt.get_cmap('viridis'), node_size=500, font_size=12)
    plt.show()

if __name__ == "__main__":

    # filepath = pickFileName(1)
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