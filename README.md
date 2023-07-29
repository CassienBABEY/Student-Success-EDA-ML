# Student-Success-EDA-ML

![image](academicsucess.jpg)

This repository contains data and code related to the exploration and prediction of student success in math and Portuguese language courses in secondary school. The data used in this project were obtained from a survey of students in these courses and contain various social, gender, and study-related information about the students.

## Data Source

The data used for this analysis was sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption?datasetId=251&sortBy=voteCount&select=student-merge.R). 

## Dataset
The dataset consists of two CSV files:

1. student-mat.csv: Data related to the math course.
2. student-por.csv: Data related to the Portuguese language course.

## Models

The `models` directory contains the saved machine learning models after training. These models are used for predicting student success based on the given features with prediction.py.

## Notebooks

- `EDA_Sandbox.ipynb`: This Jupyter Notebook includes all the explorations conducted during this project. It served as a sandbox for testing and experimenting.

- `Exploratory-Data-Analysis-of-Student-Success.ipynb`: This Jupyter Notebook contains the code and analysis for performing exploratory data analysis on the student success dataset.

- `How-to-predict-success-in-both-exams?-ML.ipynb`: This Jupyter Notebook presents the machine learning model for predicting student success in both exams based on various features.

## HTML Report

- `Exploratory-Data-Analysis-of-Student-Success.html`: This is an HTML version of the exploratory data analysis report. It contains interactive visualizations and insights from the data analysis.

- `How-to-predict-success-in-both-exams_-ML.html`: This is an HTML version of the Machine Learning Process.

## Python Scripts

- `merged.py`: This Python script provides functions for merging and preprocessing the datasets.

- `training_model.py`: This Python script handles the training of machine learning models and saves the trained models to the `models` directory.

- `prediction.py`: This Python script contains functions for making predictions using the trained machine learning models.

## Usage

To use the code and run the notebooks, you will need to install the required libraries listed below:

- pandas
- plotly
- scikit-learn
- ipywidgets (for interactive visualizations in Jupyter Notebook)
- voila (for interactive rendering of notebooks with widgets)
- matplotlib (for data visualization)
- seaborn (for data visualization)
- pickle (for model serialization)

You can install the required libraries using pip:

<code>pip install pandas plotly scikit-learn ipywidgets voila matplotlib seaborn</code>

## How to Run the Notebooks

1. Clone this repository to your local machine.
2. Install the required libraries (as mentioned above).
3. Open Jupyter Notebook in the repository directory.
4. Run the notebooks in the following order:
   - `EDA_Sandbox.ipynb`
   - `Exploratory-Data-Analysis-of-Student-Success.ipynb`
   - `How-to-predict-success-in-both-exams-ML.ipynb`

You can also explore the HTML report files, which provide a report of the data analysis.

## How to Run Python Scripts

1. Clone this repository to your local machine.
2. Install the required libraries (as mentioned above).
3. If you want to train the models and obtain their scores, run the following command in your console:  
   <code>python training_model.py data/student-mat.csv data/student-por.csv</code>
4. If you want to make a prediction for math exams, run the following command in your console:  
   <code>python prediction.py data/Student_X.csv mat</code>

   Otherwise, if you want to make a prediction for Portuguese exams, run the following command in your console:  
   <code>python prediction.py data/Student_X.csv por</code>
5. Please remember to replace the 'X' with a number between 1 and 3 (check the available data in the "data" directory).


## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

For more information about the dataset, please refer to the source provided in the Kaggle dataset:
P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance.
