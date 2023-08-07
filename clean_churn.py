#!/usr/bin/env python

""" WGU D206 Data Cleaning Performance Assessment """

import sys

# setting the random seed for reproducibility
import random
random.seed(493)

import pandas as pd # for manipulating dataframes
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for visualization
from sklearn.impute import SimpleImputer # for handling missing values

def impute_mean(df, column):
    """
    Takes a dataframe and column name and returns
    a dataframe with imputed values
    """
    mean_imputer = SimpleImputer(strategy='mean')
    df[column] = mean_imputer.fit_transform(df[column].values.reshape(-1,1))
    return df

def main():
    """Main entry point for the script."""

    # Read a csv file
    df = pd.read_csv('churn_raw_data.csv', index_col=0)
    dfx = df.copy()

    # assemble list of column names that have missing values (numerical)
    missing_columns_num = ['Children',
                    'Age',
                    'Income',
                    'Tenure',
                    'Bandwidth_GB_Year'
                    ]

    # loop over missing columns
    for col in missing_columns_num:
        dfx = impute_mean(dfx, col)

    # fill the missing values of the categorical columns with 'Unknown'
    dfx = dfx.fillna('Unknown')

    # cast column values to their correct data types
    dfx['Zip'] = dfx['Zip'].astype(str)
    dfx['Children'] = dfx['Children'].astype('int64')
    dfx['Age'] = dfx['Age'].astype('int64')

    # finding the 1st quartile
    q1 = np.quantile(dfx['MonthlyCharge'], 0.25)
    
    # finding the 3rd quartile
    q3 = np.quantile(dfx['MonthlyCharge'], 0.75)
    med = np.median(dfx['MonthlyCharge'])
    
    # finding the iqr region
    iqr = q3-q1
    
    # finding upper and lower whiskers
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    
    # filter only rows with values below the upper bound and above the lower_bound
    dfx = dfx[(dfx["MonthlyCharge"] < upper_bound) & (dfx["MonthlyCharge"] > lower_bound)]    

    dfx.to_csv('churn_cleaned_data_executable.csv', index=False)
    print('Dataframe shape: ' + str(dfx.shape))

if __name__ == '__main__':
    sys.exit(main())










__author__ = "Ednalyn C. De Dios, et al."
__copyright__ = "Copyright 2023, WGU D206 Data Cleaning Performance Assessment"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ednalyn C. De Dios"
__email__ = "ednalyn.dedios@gmail.com"
__status__ = "Prototype"