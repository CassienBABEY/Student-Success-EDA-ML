import pandas as pd
import pickle
import sys

def prediction(X, model):
    """
    Perform prediction using the specified model.

    Args:
        X (array-like or DataFrame): Input data for prediction.
        model (str): The model to use for prediction. Possible values: 'mat' or 'por'.

    Returns:
        str: Prediction result for the given student.
    """

    # Convert X to a DataFrame
    X = pd.read_csv(X)

    if model == 'mat':
        # Load the trained model for mathematics from the 'models' directory using pickle
        with open('models/Model_Classifier_Mat.pkl', 'rb') as f:
            mat_model = pickle.load(f) 
        
        # Make prediction using the loaded model
        score = mat_model.predict(X)

        # Return the prediction result
        if score == 1:
            return "The student is likely to pass the mathematics exam."
        return "The student is likely to fail the mathematics exam."
    
    if model == 'por':
        # Load the trained model for Portuguese from the 'models' directory using pickle
        with open('models/Model_Classifier_Por.pkl', 'rb') as f:
            por_model = pickle.load(f) 
        
        # Make prediction using the loaded model
        score = por_model.predict(X)

        # Return the prediction result
        if score == 1:
            return "The student is likely to pass the Portuguese exam."
        return "The student is likely to fail the Portuguese exam."
    

if __name__ == '__main__':
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) == 3:
        # Get the csv file and model name from command-line arguments
        csv_file = sys.argv[1]
        model = sys.argv[2]

        # Perform prediction using the specified model and display the result
        result = prediction(csv_file, model)
        print(result)
    else:
        print("Usage: python prediction.py <csv_file> <model>")
        print("Model can be 'mat' or 'por'")