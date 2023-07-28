# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the 'merge_dataframes' function from the 'merged.py' module
from merged import merge_dataframes

# Define a function to convert a numeric note to binary (0 if note < 10, otherwise 1)
def convert_to_binary(note):
    return 1 if note >= 10 else 0

# Define the main function for training and saving the models
def training_model(df1, df2):
    # Merge the two input DataFrames df1 and df2
    df = merge_dataframes(df1, df2)

    # Prepare the data and targets for model training
    X = df.drop(columns=['G3_mat', 'G3_por'])
    y1 = df['G3_mat'].apply(convert_to_binary)
    y2 = df['G3_por'].apply(convert_to_binary)

    # Split the data into training and testing sets for both targets
    X_train, X_test, y_train_mat, y_test_mat = train_test_split(X, y1, test_size=0.3, random_state=42)
    X_train, X_test, y_train_por, y_test_por = train_test_split(X, y2, test_size=0.3, random_state=42) 

    # Define column names for ordinal, categorical, and numerical features
    ordinal_cols = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                    'freetime', 'goout', 'Dalc', 'Walc', 'health']
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup',
                        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'Mjob',
                        'Fjob', 'reason', 'guardian']
    numerical_cols = ['age']

    # Create pipelines for ordinal, categorical, and numerical data processing
    ordinal_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse=False))

    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        StandardScaler())

    # Create a column transformer to apply the respective pipelines to different columns
    preprocess = make_column_transformer(
        (ordinal_pipeline, ordinal_cols),
        (categorical_pipeline, categorical_cols),
        (numerical_pipeline, numerical_cols),
        remainder='passthrough'
    )
    
    # Create RandomForestClassifier models with specific hyperparameters
    model_mat = make_pipeline(
        preprocess,
        RandomForestClassifier(max_depth=5,
                               min_samples_split=2,
                               min_samples_leaf=2,
                               n_estimators=500)
    )
    
    model_por = make_pipeline(
        preprocess,
        RandomForestClassifier(max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               n_estimators=100)
    )
    
    # Fit the models on the training data
    Model_Classifier_Mat = model_mat.fit(X_train, y_train_mat)
    Model_Classifier_Por = model_por.fit(X_train, y_train_por)

    # Save model scores
    Mat_Score = Model_Classifier_Mat.score(X_test, y_test_mat)
    Por_Score = Model_Classifier_Por.score(X_test, y_test_por)

    # Check if the 'models' directory exists, if not, create it
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the trained models to separate files inside the 'models' directory using pickle
    with open('models/Model_Classifier_Mat.pkl', 'wb') as f:
        pickle.dump(Model_Classifier_Mat, f)

    with open('models/Model_Classifier_Por.pkl', 'wb') as f:
        pickle.dump(Model_Classifier_Por, f)

    return f"Model mat Score: {Mat_Score}, Model por score: {Por_Score}"

# Assuming the file paths are provided as command-line arguments
if len(sys.argv) == 3:
    df1_path = sys.argv[1]
    df2_path = sys.argv[2]
    result = training_model(df1_path, df2_path)
    print(result)
else:
    print("Usage: python training_model.py dataframe_mat.csv dataframe_por.csv")