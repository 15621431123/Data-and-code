import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

file_path = r'Sheet7.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

features = df[['主室炉压', '副室炉压', '炉壁测温', '埚升速度']].values
liquid_level_temp = df['主加热功率'].values

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
X, y, scaler_features, scaler_target = preprocess_data(features, liquid_level_temp, look_back)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
mlp_model.fit(X_train_flat, y_train)

train_pred_mlp = mlp_model.predict(X_train_flat)
test_pred_mlp = mlp_model.predict(X_test_flat)

train_pred_mlp = scaler_target.inverse_transform(train_pred_mlp.reshape(-1, 1))
test_pred_mlp = scaler_target.inverse_transform(test_pred_mlp.reshape(-1, 1))

y_test_inverse = scaler_target.inverse_transform(y_test.reshape(-1, 1))
y_train_inverse = scaler_target.inverse_transform(y_train.reshape(-1, 1))

mae_train_mlp = mean_absolute_error(y_train_inverse, train_pred_mlp)
mse_train_mlp = mean_squared_error(y_train_inverse, train_pred_mlp)
r2_train_mlp = r2_score(y_train_inverse, train_pred_mlp)

mae_test_mlp = mean_absolute_error(y_test_inverse, test_pred_mlp)
mse_test_mlp = mean_squared_error(y_test_inverse, test_pred_mlp)
r2_test_mlp = r2_score(y_test_inverse, test_pred_mlp)

with pd.ExcelWriter(r'') as writer:
    train_results_mlp_df = pd.DataFrame({
        'True Values': y_train_inverse.flatten(),
        'Predictions': train_pred_mlp.flatten(),
    })
    train_results_mlp_df.to_excel(writer, sheet_name='Train Results', index=False)

    test_results_mlp_df = pd.DataFrame({
        'True Values': y_test_inverse.flatten(),
        'Predictions': test_pred_mlp.flatten(),
    })
    test_results_mlp_df.to_excel(writer, sheet_name='Test Results', index=False)

    metrics_mlp_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'R²'],
        'Train': [mae_train_mlp, mse_train_mlp, r2_train_mlp],
        'Test': [mae_test_mlp, mse_test_mlp, r2_test_mlp],
    })
    metrics_mlp_df.to_excel(writer, sheet_name='Metrics', index=False)

print("MLP Regressor with sliding window results and metrics saved to Excel.")