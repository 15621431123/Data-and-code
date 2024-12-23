import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

file_path = r'Sheet7_reversed.xlsx'
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
        X.append(scaled_features[i:i + look_back].flatten())  # 扁平化
        y.append(scaled_target[i + look_back, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler_features, scaler_target

look_back = 8
X, y, scaler_features, scaler_target = preprocess_data(features, liquid_level_temp, look_back)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

svr_model = SVR(kernel='rbf', C=60, gamma='scale')

svr_model.fit(X_train, y_train)

train_pred = svr_model.predict(X_train)
test_pred = svr_model.predict(X_test)

train_pred = scaler_target.inverse_transform(train_pred.reshape(-1, 1))
test_pred = scaler_target.inverse_transform(test_pred.reshape(-1, 1))

y_test_inverse = scaler_target.inverse_transform(y_test.reshape(-1, 1))
y_train_inverse = scaler_target.inverse_transform(y_train.reshape(-1, 1))

mae_train = mean_absolute_error(y_train_inverse, train_pred)
mse_train = mean_squared_error(y_train_inverse, train_pred)
r2_train = r2_score(y_train_inverse, train_pred)

mae_test = mean_absolute_error(y_test_inverse, test_pred)
mse_test = mean_squared_error(y_test_inverse, test_pred)
r2_test = r2_score(y_test_inverse, test_pred)

with pd.ExcelWriter(r'.xlsx') as writer:
    train_results_df = pd.DataFrame({
        'True Values': y_train_inverse.flatten(),
        'Predictions': train_pred.flatten(),
    })
    train_results_df.to_excel(writer, sheet_name='Train Results', index=False)

    test_results_df = pd.DataFrame({
        'True Values': y_test_inverse.flatten(),
        'Predictions': test_pred.flatten(),
    })
    test_results_df.to_excel(writer, sheet_name='Test Results', index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'R²'],
        'Train': [mae_train, mse_train, r2_train],
        'Test': [mae_test, mse_test, r2_test],
    })
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

print("SVR results and metrics saved to Excel.")