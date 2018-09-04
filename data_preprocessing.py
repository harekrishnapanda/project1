# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 01:34:02 2018

@author: Harekrishna
"""

# Data Cleaning
def data_cleaning(df):
    df.info()
    df.isnull().sum()
    # dropping the NAN values as the count of NAN records is aroung 3/4% of total records
    df = df.dropna()
    #df.attacking_work_rate.unique()
    df.attacking_work_rate.value_counts()
    awr = ['medium', 'high','low']
    # removing the non related categorical values as their count is vey less but might impact our model
    df = df[df['attacking_work_rate'].isin(awr)]
    df.defensive_work_rate.value_counts()
    df = df[df['defensive_work_rate'].isin(awr)]
    return(df)
