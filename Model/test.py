import sys
import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.optimize import AdamOptimizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('Dataset/dataset.csv')

# Clean the data
def clean_data(data):
    data = data[~(data == 'Screen off (locked)').any(axis=1)]
    data = data[~(data == 'Screen on (unlocked)').any(axis=1)]
    data = data[~(data == 'Screen off (unlocked)').any(axis=1)]
    data = data[~(data == 'Screen on (locked)').any(axis=1)]
    data = data[~(data == 'Screen on').any(axis=1)]
    data = data[~(data == 'Screen off').any(axis=1)]
    data = data[~(data == 'Device shutdown').any(axis=1)]
    data = data[~(data == 'Device boot').any(axis=1)]
    data = data.dropna()
    data.index = range(len(data))
    return data

# Encode the data
def encode_data(data):
    label_encoder_app = LabelEncoder()
    encoded_data = label_encoder_app.fit_transform(data.iloc[:, 0])
    encoded_data = pd.DataFrame(data=encoded_data)
    return [encoded_data, label_encoder_app]

# Split the data into training and testing sets
def split_into_train_test_set(encoded_data):
    train_set = encoded_data.iloc[:1901, 0].values
    test_set = encoded_data.iloc[1901:, 0].values
    return [train_set, test_set]

data = clean_data(data)
encoded_data, label_encoder_app = encode_data(data)
train_set, test_set = split_into_train_test_set(encoded_data)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(train_set.reshape(-1, 1))

# Prepare training data
X_train = []
y_train = []

for i in range(10, 1901):
    X_train.append(training_set_scaled[i-10:i, 0])
    y_train.append(train_set[i])

X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

label_encoder_y = LabelEncoder()
y_train = label_encoder_y.fit_transform(y_train)
y_train = np.array(y_train)
y_train = to_categorical(y_train, num_classes=36)

# Define quantum device and circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    AngleEmbedding(inputs, wires=range(n_qubits))
    StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define qLSTM cell
class qLSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = pnp.random.random(size=(input_dim + hidden_dim, 3, hidden_dim))

    def forward(self, x_t, h_prev):
        z_t = np.concatenate([x_t, h_prev])
        h_t = np.tanh(np.dot(z_t, self.weights[:, 0, :]))
        c_t = np.tanh(np.dot(z_t, self.weights[:, 1, :]))
        o_t = np.tanh(np.dot(z_t, self.weights[:, 2, :]))
        h_next = o_t * np.tanh(c_t)
        return h_next, c_t

# Define the model
class qLSTMModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.qLSTM = qLSTMCell(input_dim, hidden_dim)
        self.dense_weights = pnp.random.random(size=(hidden_dim, output_dim))

    def forward(self, x):
        h_t, c_t = np.zeros(self.qLSTM.hidden_dim), np.zeros(self.qLSTM.hidden_dim)
        for t in range(x.shape[0]):
            h_t, c_t = self.qLSTM.forward(x[t], h_t)
        output = np.dot(h_t, self.dense_weights)
        return output

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)

# Instantiate and train the model
input_dim = X_train.shape[2]
hidden_dim = 4  # Number of qubits
output_dim = 36

model = qLSTMModel(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
def loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

optimizer = AdamOptimizer(0.01)
epochs = 150

# Training loop
for epoch in range(epochs):
    for i in range(len(X_train)):
        x, y = X_train[i], y_train[i]
        y_pred = model.forward(x)
        model.qLSTM.weights = optimizer.step(lambda w: loss(y, y_pred), model.qLSTM.weights)

    if epoch % 10 == 0:
        train_loss = np.mean([loss(y_train[i], model.forward(X_train[i])) for i in range(len(X_train))])
        print(f"Epoch {epoch}: Loss {train_loss}")

# Testing
total_dataset = encoded_data.iloc[:, 0]
inputs = total_dataset[len(total_dataset) - len(test_set) - 10:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test = []

for i in range(10, 397):
    X_test.append(inputs[i-10:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_app = model.predict(X_test)

# Confusion matrix
cm = np.zeros(shape=(2, 2))
for i in range(387):
    if test_set[i] == predicted_app[i]:
        cm[1, 1] += 1
    else:
        cm[1, 0] += 1

# Final predictions
idx = (-predicted_app).argsort()
idx = pd.DataFrame(idx)
prediction = label_encoder_app.inverse_transform(idx.iloc[:, 0])
prediction = pd.DataFrame(data=prediction)
actual_app_used = label_encoder_app.inverse_transform(test_set)
actual_app_used = pd.DataFrame(data=actual_app_used)

for i in range(1, 4):
    idx3 = label_encoder_app.inverse_transform(idx.iloc[:, i:i+1])
    idx3 = pd.DataFrame(data=idx3)
    prediction = pd.concat([prediction, idx3], ignore_index=True, axis=1)

final_outcome = pd.concat([prediction, actual_app_used], axis=1)
final_outcome.columns = ['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4', 'Actual App Used']

print('***********************************FINAL PREDICTION*********************************')
print(final_outcome)
