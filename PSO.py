import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

file_path = r'.xlsx'
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

    return X, y, scaler_target

look_back = 8
X, y, scaler_target = preprocess_data(features, liquid_level_temp, look_back)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def create_model(input_shape, neuron_count, learning_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(neuron_count, activation='tanh'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def evaluate_model(model, X_val, y_val, scaler_target):
    y_pred = model.predict(X_val)

    y_pred = scaler_target.inverse_transform(y_pred)
    y_val = scaler_target.inverse_transform(y_val.reshape(-1, 1))

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    return y_pred, y_val, mse, mae

def fitness(particle):
    model = create_model((look_back, features.shape[1]), int(particle[0]), particle[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model_pso.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=0)

    model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_data=(X_val, y_val),
              callbacks=[checkpoint, early_stopping])
    model = load_model('best_model_pso.keras')

    _, _, mse, mae = evaluate_model(model, X_val, y_val, scaler_target)

    tf.keras.backend.clear_session()
    return mse

n_particles = 20
n_dimensions = 2
n_iterations =100
c1 = 2
c2 = 2

neuron_range = (10, 400)
learning_rate_range = (0.00001, 0.1)

particles = np.zeros((n_particles, n_dimensions))
velocities = np.zeros((n_particles, n_dimensions))

particles[:, 0] = np.random.uniform(neuron_range[0], neuron_range[1], n_particles)
particles[:, 1] = np.random.uniform(learning_rate_range[0], learning_rate_range[1], n_particles)

pbest = np.copy(particles)
pbest_scores = np.array([float('inf')] * n_particles)
gbest = None
gbest_score = float('inf')

iteration_info = []

for iteration in range(n_iterations):
    for i in range(n_particles):
        score = fitness(particles[i])

        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest[i] = particles[i]

        if score < gbest_score:
            gbest_score = score
            gbest = particles[i]

    Wmax = 0.9
    Wmin = 0.4
    w = Wmax - (Wmax - Wmin) * (iteration / n_iterations)
    for i in range(n_particles):
        r1 = np.random.rand(n_dimensions)
        r2 = np.random.rand(n_dimensions)

        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest[i] - particles[i]) +
                         c2 * r2 * (gbest - particles[i]))

        velocities[i][0] = np.clip(velocities[i][0], -0.2 * (neuron_range[1] - neuron_range[0]), 0.2 * (neuron_range[1] - neuron_range[0]))
        velocities[i][1] = np.clip(velocities[i][1], -0.2 * (learning_rate_range[1] - learning_rate_range[0]), 0.2 * (learning_rate_range[1] - learning_rate_range[0]))

        particles[i] += velocities[i]

        particles[i][0] = np.clip(particles[i][0], neuron_range[0], neuron_range[1])
        particles[i][1] = np.clip(particles[i][1], learning_rate_range[0], learning_rate_range[1])

    iteration_info.append((iteration, gbest_score, gbest))

    # 输出当前迭代信息
    print("Iteration: {}, Best Score: {:.4f}, Best Parameters: Neurons = {}, Learning Rate = {:.4f}".format(
        iteration, gbest_score, int(gbest[0]), gbest[1]))

print("Best parameters: Neurons = {}, Learning Rate = {}".format(int(gbest[0]), gbest[1]))

final_model = create_model((look_back, features.shape[1]), int(gbest[0]), gbest[1])
checkpoint = ModelCheckpoint('best_model_final.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

final_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2, validation_data=(X_val, y_val),
                callbacks=[checkpoint, early_stopping])

train_pred = final_model.predict(X_train)
test_pred = final_model.predict(X_test)

train_pred = scaler_target.inverse_transform(train_pred)
test_pred = scaler_target.inverse_transform(test_pred)

y_test_inverse = scaler_target.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_inverse, test_pred)
mse = mean_squared_error(y_test_inverse, test_pred)

print("Evaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)

output_df = pd.DataFrame(iteration_info, columns=['Iteration', 'Best Score', 'Best Parameters'])
output_df.to_excel(r'', index=False)
