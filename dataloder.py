import pandas as pd
from sklearn.preprocessing import StandardScaler

class dataloder():
    def __init__(self):
        pass
    def get_train_data(self):
        path = "train_dataset.csv"
        train_df = pd.read_csv(path)
        X_train = train_df[["x1", "x2"]].values
        y_train = train_df["label"].values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        return X_train_scaled, y_train
    
    def get_test_data(self):
        path = "test_dataset.csv"
        test_df = pd.read_csv(path)
        X_test = test_df[["x1", "x2"]].values
        y_test = test_df["label"].values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_test)
        
        return X_train_scaled, y_test