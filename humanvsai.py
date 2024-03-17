import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

def get_sample_rate(file):
    try:
        # Load audio file using librosa
        audio, sr = librosa.load(file, sr=None, duration=3)
        return sr
    except Exception as e:
        print(f"Error getting sample rate for {file}: {str(e)}")
        return None

    
# Function to preprocess audio data
def preprocess_audio(audio_files, labels, sampling_rate=22050, duration=3, max_len=None):
    X = []
    y = []
    actual_max_len = 0  # Initialize max_len
    for file, label in zip(audio_files, labels):
        try:
            print(f"Processing {file}")
            sr = get_sample_rate(file)
            if sr is not None:
                audio, _ = librosa.load(file, sr=sr, duration=duration, res_type='kaiser_fast')
                features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
                actual_max_len = max(actual_max_len, features.shape[1])  # Update actual_max_len
                X.append(features)
                y.append(label)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    # Pad or trim features to max_len
    if max_len is None:
        max_len = actual_max_len
    for i in range(len(X)):
        if X[i].shape[1] < max_len:
            padded_features = np.pad(X[i], ((0, 0), (0, max_len - X[i].shape[1])), 'constant')
        else:
            padded_features = X[i][:, :max_len]  # Trim features if longer than max_len
        X[i] = padded_features[:,:,np.newaxis]  # Add channel dimension
    return np.array(X), np.array(y), actual_max_len


dataset_folder = "Dataset"

# Initialize lists to store file paths and corresponding labels
audio_files = []
labels = []

# Define folder names
folders = ["Human", "AI"]

# Assign labels for each folder
for label, folder in enumerate(folders):
    # Get the full path to the current folder
    folder_path = os.path.join(dataset_folder, folder)
    # Ensure the folder exists
    if os.path.exists(folder_path):
        # Walk through all files in the current folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Check if the file is an audio file (MP3, WAV, etc.)
                if file.endswith(('.mp3', '.wav','.flac')):
                    # Add the file path to the list of audio files
                    audio_files.append(os.path.join(root, file))
                    # Assign label to the file
                    labels.append(label)

# Preprocess data
X, y, max_len = preprocess_audio(audio_files, labels)

# Check if X is empty or not
if len(X) == 0:
    raise ValueError("Dataset does not contain any samples.")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(20, max_len, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy}')

# Predict probabilities on new data
def predict_probabilities(audio_files):
    if len(audio_files) == 0:
        print("No samples for prediction.")
        return
    X_new, _, _ = preprocess_audio(audio_files, np.zeros(len(audio_files)), max_len=max_len)
    predictions = model.predict(X_new)
    for i, prob in enumerate(predictions):
        print(f"Probability of AI-generated voice for sample {i+1}: {prob[0]}")
    return X_new

folder_path = "Dataset/Test"

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter out only the audio files (MP3, WAV, etc.)
new_test_audio_files = [os.path.join(folder_path, file) for file in files if file.endswith(('.mp3', '.wav','.flac'))] 
# Add paths to new test audio files
X_test=predict_probabilities(new_test_audio_files)

# Predict labels for test data
Y_pred = model.predict(X_test)

# Convert probabilities to binary predictions
Y_pred_binary = np.round(Y_pred).astype(int)

# Evaluate test accuracy
test_accuracy = np.mean(Y_pred_binary)
print("Test Accuracy:", test_accuracy)