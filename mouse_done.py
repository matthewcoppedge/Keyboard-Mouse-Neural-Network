import os
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import matplotlib.pyplot as plt

# Function to preprocess mouse data
def preprocess_mouse_data(mouse_df, max_sequence_length=300):
    mouse_df['time_seconds'] = pd.to_datetime(mouse_df['time']).dt.second
    mouse_df['time_milliseconds'] = pd.to_datetime(mouse_df['time']).dt.microsecond // 1000
    mouse_features = mouse_df[['rX', 'rY', 'pX', 'pY', 'time_seconds', 'time_milliseconds']].values

    # Pad sequences with zeros if shorter than max_sequence_length
    mouse_features_padded = pad_sequences([mouse_features], maxlen=max_sequence_length, padding='post', truncating='post')[0]

    return mouse_features_padded.reshape((1, mouse_features_padded.shape[0], mouse_features_padded.shape[1]))

# Directory containing user folders
base_directory = 'BB-MAS_Dataset'
model_directory = 'mouse_user_model'  

# Initialize lists to store features + labels
all_mouse_features = []
all_labels = []
user_models = {}  # Dictionary to store individual models

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Iterate through user folders
for user_folder in os.listdir(base_directory):
    user_path = os.path.join(base_directory, user_folder)

    # Check if it's a directory
    if os.path.isdir(user_path):
        # Extract user number from the folder name
        user_number = int(user_folder)

        # Load and preprocess mouse data
        mouse_file_path = os.path.join(user_path, f'{user_number}_Mouse_Move.csv')
        mouse_df = pd.read_csv(mouse_file_path)

        
        if len(mouse_df) >= 2:
            # Split the data into training and validation sets
            train_data, val_data = train_test_split(mouse_df, test_size=0.2, random_state=42)

            # Preprocess and store features and labels for training data
            train_mouse_features = preprocess_mouse_data(train_data)
            all_mouse_features.append(train_mouse_features)
            all_labels.append(user_number)

            # Preprocess and store features and labels for validation data
            val_mouse_features = preprocess_mouse_data(val_data)
            all_mouse_features.append(val_mouse_features)
            all_labels.append(user_number)

# Convert lists to NumPy arrays
X_mouse = np.concatenate(all_mouse_features, axis=0)
y = np.array(all_labels)

# Convert user labels to categorical
y_categorical = to_categorical(y)

# Iterate through user models
for user_number in np.unique(y):


    # Find indices of current user 
    user_indices = np.where(y == user_number)[0]

    # Extract data for current user
    user_X = X_mouse[user_indices]
    user_y = y_categorical[user_indices]

    X_train, X_val, y_train, y_val = train_test_split(user_X, user_y, test_size=0.2, random_state=42)

    # Build the model for mouse data using 1D-CNN
    input_mouse = Input(shape=(user_X.shape[1], user_X.shape[2]))

    # 1D CNN + RNN for Mouse 
    x_mouse = Conv1D(32, kernel_size=3, activation='relu')(input_mouse)
    x_mouse = MaxPooling1D(pool_size=2)(x_mouse)
    x_mouse = LSTM(64, return_sequences=True)(x_mouse)
    x_mouse = LSTM(64)(x_mouse)

    # Fully Connected Layers w/Dropout for regularization
    x_mouse = Dense(128, activation='relu')(x_mouse)
    x_mouse = Dropout(0.5)(x_mouse)
    x_mouse = Flatten()(x_mouse)

    # Output Layer
    num_users = len(np.unique(y))  # For individual models, each model predicts for a single user
    output_layer = Dense(num_users + 1, activation='softmax')(x_mouse)

    # Create model 
    user_model = Model(inputs=input_mouse, outputs=output_layer)

    # Compile model
    user_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    log_dir = f'log_mouse/user_{user_number}'

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train model
    user_model.fit(
        X_train, y_train,
        epochs=10, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, tensorboard_callback]
    )

    # Save individual model 
    model_filename = f'user_{user_number}_model.tf'
    model_path = os.path.join(model_directory, model_filename)
    user_model.save(model_path)

    # Store the model in dictionary
    user_models[user_number] = model_path

    tf.keras.backend.clear_session() #important

# The rest of your code for making predictions and plotting the bar chart
claimed_user_number = 45  # Replace with the claimed user number
new_data_file_path = '#45_genuine_mouse.csv'  # Replace with the path to your new data file

# Load the model for the claimed user
model_directory = 'mouse_user_model'
model_path = f'{model_directory}/user_{claimed_user_number}_model.tf'
user_model = tf.keras.models.load_model(model_path)

# Load and preprocess the new data
new_mouse_df = pd.read_csv(new_data_file_path)
X_new_mouse = preprocess_mouse_data(new_mouse_df)  # Assuming you have a preprocess_mouse_data function

# Make predictions using loaded model
predictions_new_data = user_model.predict(X_new_mouse)

# Interpret predictions
predicted_user_probability = predictions_new_data[0, claimed_user_number]  # Adjust the index if needed

# Get predicted probabilities for all users
all_predicted_probabilities = predictions_new_data[0, :]

# Plot the bar chart
users = range(1, len(all_predicted_probabilities) + 1)
plt.bar(users, all_predicted_probabilities, color='blue')
plt.xlabel('User Number')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probabilities for Each User')
plt.xticks(users)
plt.show()

# Set a threshold for confidence
threshold = 0.5
predicted_user = claimed_user_number if predicted_user_probability >= threshold else -1  # -1 indicates not confident

# Print the results
print(f"Claimed User: {claimed_user_number}")
print(f"Predicted User: {predicted_user}")
print(f"Predicted Probability: {predicted_user_probability}")

