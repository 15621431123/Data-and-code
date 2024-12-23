import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

model_path = r'C:\best_model_final.keras'
loaded_model = load_model(model_path)

file_path = r'Sheet6_reversed.xlsx'
df_new = pd.read_excel(file_path, sheet_name='Sheet1')

def preprocess_data(features, target, look_back):
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_features) - look_back):
        X.append(scaled_features[i:i + look_back])
        y.append(scaled_target[i + look_back, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler_features, scaler_target

look_back = 8
features_new = df_new[['熔接流明', '炉壁测温', '主室炉压', '副室炉压', '晶升位置', '主加热功率']].values
liquid_level_temp_new = df_new['温度'].values

X_new, y_new, scaler_features_new, scaler_target_new = preprocess_data(features_new, liquid_level_temp_new, look_back)

new_predictions = loaded_model.predict(X_new)

new_predictions = scaler_target_new.inverse_transform(new_predictions)
y_new_inverse = scaler_target_new.inverse_transform(y_new.reshape(-1, 1))

mae = mean_absolute_error(y_new_inverse, new_predictions)
mse = mean_squared_error(y_new_inverse, new_predictions)
r2 = r2_score(y_new_inverse, new_predictions)

with pd.ExcelWriter(r'.xlsx') as writer:
    results_df = pd.DataFrame({
        'True Values': y_new_inverse.flatten(),
        'Predictions': new_predictions.flatten(),
    })
    results_df.to_excel(writer, sheet_name='Results', index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'R²'],
        'Value': [mae, mse, r2],
    })
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

print("Results and metrics for new data saved to Excel.")