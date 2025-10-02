import pandas as pd 
from sklearn.dataset import load_digits

def load_data():
    """A function to load data from the sklearn dataset library"""
    digits = load_digits() 
    df = pd.DataFrame(digits.data, columns=digits.feature_names)
    df['target'] = digits.target # define the target 
    df['target desc'] = df.target.apply(lambda x : digits.target_names[x]) # applying the target labels
    
    return digits, df