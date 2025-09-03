from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def scale_data(df, scaler_type="StandardScaler"):
    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    return scaled_df, scaler
