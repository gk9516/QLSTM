import sys
import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.optimize import AdamOptimizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('Dataset/dataset.csv')

# Clean the data
def clean_data(data):
    # Remove specific entries that are not useful for prediction
    remove_strings = ['Screen off (locked)', 'Screen on (unlocked)', 'Screen off (unlocked)',
                      'Screen on (locked)', 'Screen on', 'Screen off', 'Device shutdown', 'Device boot']
    for entry in remove_strings:
        data = data[~data.iloc[:, 0].str.contains(entry, na=False)]

    data = data.dropna().reset_index(drop=True)
    return data

# Encode the data
def encode_data(data):
    label_encoder_app = LabelEncoder()
    data['Encoded_App'] = label_encoder_app.fit_transform(data.iloc[:, 0])
    return data, label_encoder_app

# Split the data into training and testing sets
def split_into_train_test_set(encoded_data):
    train_set, test_set = train_test_split(encoded_data, test_size=0.2, shuffle=False)
    return train_set, test_set

# Scale the data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

# Prepare training data
def prepare_training_data(train_set, sequence_length):
    X_train, y_train = [], []
    for i in range(sequence_length, len(train_set)):
        X_train.append(train_set[i-sequence_length:i, 0])
        y_train.append(train_set[i, 1])
    return np.array(X_train), np.array(y_train)

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
        x_t = x_t.flatten()
        h_prev = h_prev.flatten()
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
        y_pred = []
        for sample in x:
            output = self.forward(sample)
            y_pred.append(np.argmax(output, axis=0))
        return np.array(y_pred)

# Instantiate and train the model
def train_model(model, X_train, y_train, optimizer, epochs):
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x, y = X_train[i], y_train[i]
            y_pred = model.forward(x)
            model.qLSTM.weights = optimizer.step(lambda w: loss(y, y_pred), model.qLSTM.weights)

        if epoch % 10 == 0:
            train_loss = np.mean([loss(y_train[i], model.forward(X_train[i])) for i in range(len(X_train))])
            print(f"Epoch {epoch}: Loss {train_loss}")

# Define the loss function
def loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Testing Phase
def test_model(model, X_test, scaler, label_encoder):
    inputs = scaler.transform(X_test)
    X_test = []
    for i in range(len(inputs)):
        X_test.append(inputs[i:i+10, 0])

    X_test = np.array(X_test).reshape(-1, 10, 1)
    predicted_app = model.predict(X_test)
    predicted_app = np.clip(predicted_app, 0, len(label_encoder.classes_) - 1)

    idx = (-predicted_app).argsort()
    idx_clipped = np.clip(idx, 0, len(label_encoder.classes_) - 1)
    prediction = label_encoder.inverse_transform(idx_clipped.flatten())

    actual_app_used = label_encoder.inverse_transform(test_set['Encoded_App'].values)

    final_outcome = pd.DataFrame({'Prediction': prediction, 'Actual App Used': actual_app_used})
    return final_outcome

# Main function
def main():
    # Load and clean the data
    data = clean_data(data)

    # Encode the data
    encoded_data, label_encoder = encode_data(data)

    # Split into training and testing sets
    train_set, test_set = split_into_train_test_set(encoded_data)

    # Scale the data
    scaled_train_data, scaler = scale_data(train_set['Encoded_App'])

    # Prepare training data
    X_train, y_train = prepare_training_data(scaled_train_data, sequence_length=10)

    # Instantiate the model
    input_dim = X_train.shape[1]
    hidden_dim = 4  # Number of qubits
    output_dim = len(label_encoder.classes_)
    model = qLSTMModel(input_dim, hidden_dim, output_dim)

    # Define the optimizer and train the model
    optimizer = AdamOptimizer(0.01)
    epochs = 150
    train_model(model, X_train, y_train, optimizer, epochs)

    # Prepare testing data and evaluate the model
    X_test = test_set['Encoded_App'].values
    final_results = test_model(model, X_test, scaler, label_encoder)

    print('***********************************FINAL PREDICTION*********************************')
    print(final_results)

if __name__ == "__main__":
    main()
