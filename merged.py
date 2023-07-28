import pandas as pd

def merge_dataframes(df1, df2):

    '''
Merge our dataframes.
Final dataframe got 382 students.
We removed the columns G1 and G2 as they were highly correlated with the target variable. 
Additionally, we dropped the 'absences' column due to a large number of missing values.

'''

    df1 = pd.read_csv(df1)
    df2 = pd.read_csv(df2)
    # Columns used for merging
    keys_to_merge = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu',
        'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet'
    ]

    # Merge DataFrames using the merge keys and adding suffixes to columns in case of duplicates
    merged_df = df1.merge(df2, on=keys_to_merge, suffixes=('_mat', '_por'))

    # Get the list of columns that contain '_mat' and '_por'
    matporcol = [
        col for col in merged_df.columns if '_mat' in col or '_por' in col
    ]

    # Remove the suffixes and then duplicates to obtain unique column names
    cols_unique = merged_df[matporcol].columns.str.replace(
        '_mat', '').str.replace('_por', '').drop_duplicates()

    # Remove exam results as they are different for the two datasets
    cols_unique = cols_unique.drop(cols_unique[-3:])

    # Create a list of column pairs to merge similar columns
    column_pairs = [(col + '_mat', col + '_por') for col in cols_unique]

    # Loop to merge columns in similar pairs
    for col_mat, col_por in column_pairs:
        merged_df[col_mat] = merged_df[col_mat].where(
            merged_df[col_mat] == merged_df[col_por])

    # Remove the '_y' columns after merging if needed
    for _, col_por in column_pairs:
        merged_df.drop(columns=[col_por], inplace=True)

    merged_df = merged_df.drop(columns=['absences_mat', 'G1_mat', 'G2_mat', 'G1_por','G2_por'])

    columns_to_exclude = ['G3_mat', 'G3_por']
    columns_to_rename = [col for col in merged_df.columns if col not in columns_to_exclude]

    for col in columns_to_rename:
        if col.endswith('_mat'):
            new_col_name = col.replace('_mat', '')
            merged_df.rename(columns={col: new_col_name}, inplace=True)

    return merged_df