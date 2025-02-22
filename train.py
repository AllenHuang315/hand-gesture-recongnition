import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Load the dataset
data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check original shape
print("Original data shape:", data.shape)  # Should print (268, 42)

# Dynamically calculate the shape for LSTM input
total_elements = data.size  # Total number of elements in the data
num_samples = data.shape[0]  # Actual number of samples (268 in this case)

# Calculate the total number of features (timesteps * features) based on the total elements
elements_per_sample = total_elements // num_samples

# Find a factorization of elements_per_sample that makes sense for timesteps and features
# We will assume that timesteps <= 10 and features will be the remainder
timesteps = 6  # Chosen arbitrary timestep, you can adjust this
features = elements_per_sample // timesteps

print(f"Reshaping data into ({num_samples}, {timesteps}, {features})")

# Reshape the data into (samples, timesteps, features)
data = data.reshape((num_samples, timesteps, features))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert labels to categorical
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy: {:.2f}%'.format(accuracy * 100))

# Save the model
model.save('lstm_model.h5')
