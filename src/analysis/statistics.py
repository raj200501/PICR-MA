import pandas as pd

def mean_se(df,col):
    return df[col].mean(), df[col].std(ddof=1)/(len(df)**0.5 if len(df)>1 else 1.0)
