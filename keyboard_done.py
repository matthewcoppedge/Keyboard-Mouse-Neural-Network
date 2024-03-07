import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine

# Function to preprocess keyboard data
def preprocess_keyboard_data(keyboard_df, max_sequence_length=300):
	keyboard_df['time_seconds'] = pd.to_datetime(keyboard_df['time']).dt.second
	key_encoder = LabelEncoder()
	keyboard_df['key_encoded'] = key_encoder.fit_transform(keyboard_df['key'])
	keyboard_features = keyboard_df[['time_seconds', 'key_encoded', 'direction']].values

	# Pad sequences with zeros if shorter than max_sequence_length
	keyboard_features_padded = pad_sequences([keyboard_features], maxlen=max_sequence_length, padding='post', truncating='post')[0]

	return keyboard_features_padded.reshape((1, keyboard_features_padded.shape[0], keyboard_features_padded.shape[1]))

# Directory containing the generated user folders for neural networks
base_directory = 'BB-MAS_Dataset'
model_directory = 'keyboard_user_model' 

# Initialize lists to store features + labels
all_keyboard_features = []
all_labels = []
user_models = {}  # Dictionary to store individual models

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Iterate through all of the user folders
for user_folder in os.listdir(base_directory):
	user_path = os.path.join(base_directory, user_folder)

	# Check if it's a directory
	if os.path.isdir(user_path):
		# Extract user number from the folder name
		user_number = int(user_folder)

		# Load and preprocess keyboard data
		keyboard_file_path = os.path.join(user_path, f'{user_number}_Desktop_Keyboard.csv')
		keyboard_df = pd.read_csv(keyboard_file_path)

		# Assuming you have at least 2 data points for splitting
		if len(keyboard_df) >= 2:
			# Split the data into training and validation sets
			train_data, val_data = train_test_split(keyboard_df, test_size=0.2, random_state=42)

			# Preprocess and store features and labels for training data
			train_keyboard_features = preprocess_keyboard_data(train_data)
			all_keyboard_features.append(train_keyboard_features)
			all_labels.append(user_number)

			# Preprocess and store features and labels for validation data
			val_keyboard_features = preprocess_keyboard_data(val_data)
			all_keyboard_features.append(val_keyboard_features)
			all_labels.append(user_number)

# Convert lists to NumPy arrays to make it all work
X_keyboard = np.concatenate(all_keyboard_features, axis=0)
y = np.array(all_labels)

# Convert user labels to categorical (necessary)
y_categorical = to_categorical(y)

# Iterate through user models
for user_number in np.unique(y):
	# Find indice of current user
	user_indices = np.where(y == user_number)[0]

	# Extract data for user
	user_X = X_keyboard[user_indices]
	user_y = y_categorical[user_indices]

	X_train, X_val, y_train, y_val = train_test_split(user_X, user_y, test_size=0.2, random_state=42)

	# Build the model
	input_keyboard = Input(shape=(user_X.shape[1], user_X.shape[2]), name='input_keyboard')

	# 1D CNN + RNN 
	x_keyboard = Conv1D(32, kernel_size=3, activation='relu', name='conv1d_keyboard')(input_keyboard)
	x_keyboard = MaxPooling1D(pool_size=2, name='maxpool_keyboard')(x_keyboard)
	x_keyboard = LSTM(64, return_sequences=True, name='lstm1_keyboard')(x_keyboard)
	x_keyboard = LSTM(64, name='lstm2_keyboard')(x_keyboard)

	# Fully Connected Layers +  Dropout for regularization (helps to prevent overfitting of data)
	x_keyboard = Dense(128, activation='relu', name='dense_keyboard')(x_keyboard)
	x_keyboard = Dropout(0.5, name='dropout_keyboard')(x_keyboard)
	x_keyboard = Flatten(name='flatten_keyboard')(x_keyboard)

	# Output Layer
	num_users = len(np.unique(y))  # For individual models, each model predicts for a single user
	output_layer = Dense(num_users + 1, activation='softmax', name='output_keyboard')(x_keyboard)

	# Create the model
	user_model = Model(inputs=input_keyboard, outputs=output_layer, name=f'user_model_keyboard_{user_number}')

	# Compile  model
	user_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  

	log_dir = f'log_keyboard/user_{user_number}'  
	tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)  

	# Train model
	user_model.fit(
			X_train, y_train,
			epochs=25, batch_size=64,
			validation_data=(X_val, y_val),
			callbacks=[early_stopping, tensorboard_callback]
			)

	# Save individual model
	model_filename = f'user_{user_number}_model.tf'
	model_path = os.path.join(model_directory, model_filename)
	user_model.save(model_path)

	# Store the model
	user_models[user_number] = model_path

	# Clear session after each user, makes it so that neural networks don't grow to enormous sizes
	tf.keras.backend.clear_session()

claimed_user_number = 45  # Replace w/different number when necessary for testing

new_data_file_path = '#45_genuine_keyboard.csv'  #Simulate new data that is claiming to belong to user specified above

# Load the model for claimed user
model_directory = 'keyboard_user_model'
model_path = f'{model_directory}/user_{claimed_user_number}_model.tf'
user_model = tf.keras.models.load_model(model_path)

# Load and preprocess new data
new_keyboard_df = pd.read_csv(new_data_file_path)
X_new_keyboard = preprocess_keyboard_data(new_keyboard_df) 

# Make predictions using the loaded model
predictions_new_data = user_model.predict(X_new_keyboard)

# Interpret predictions
predicted_user_probability = predictions_new_data[0, claimed_user_number]  

# Confidence threshold (for normal testing, at least 50% is required)
threshold = 0.5
predicted_user = claimed_user_number if predicted_user_probability >= threshold else -1  # -1 indicates not confident'

# Output Results of Testing
print(f"Claimed User: {claimed_user_number}")
print(f"Predicted User: {predicted_user}")
print(f"Predicted Probability: {predicted_user_probability}")

