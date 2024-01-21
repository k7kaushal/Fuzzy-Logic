import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model

# Generate synthetic data
data = "parkinsons.data"
data = pd.read_csv(data)
print(data.head())
print(data.shape[1])
X = np.hstack((data.iloc[:, 1:17].values, data.iloc[:, 18:].values))
y = data.iloc[:, 17].values 
print(y[:5], X[:5, :])

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size].astype(np.float32), X[train_size:].astype(np.float32)
y_train, y_test = y[:train_size].astype(np.float32), y[train_size:].astype(np.float32)

# Custom Fuzzy Layer
class FuzzyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
    def call(self, x):
        return tf.math.sigmoid(tf.matmul(x, self.kernel))

# Neural Network Model
input_layer = Input(shape=(22,))
fuzzy_layer = FuzzyLayer(10)(input_layer)
hidden_layer = Dense(10, activation='relu')(fuzzy_layer)
output_layer = Dense(1)(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Evaluate the model
predictions = model.predict(X_test)
mse = np.mean((y_test - predictions.flatten())**2)
print("Mean Squared Error:", mse)

# Plotting results
plt.scatter(X_test[:,0], y_test, label='True')
plt.scatter(X_test[:,0], predictions, label='Predicted')
plt.legend()
plt.title("Fuzzy Neural Network Predictions")
plt.show()