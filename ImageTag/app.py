import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Path to the dataset folder
dataset_path = "Dataset"

# Function to extract color features from an image
def extract_features(image_path):
    img = Image.open(image_path).resize((100, 100))  # Resize to 100x100
    img_array = np.array(img)  # Convert to array
    # Mean of red, green, and blue (RGB) colors
    red_mean = np.mean(img_array[:, :, 0])
    green_mean = np.mean(img_array[:, :, 1])
    blue_mean = np.mean(img_array[:, :, 2])
    return [red_mean, green_mean, blue_mean]

# Collect data
data = []
labels = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        full_path = os.path.join(dataset_path, filename)
        features = extract_features(full_path)
        data.append(features)
        # Labeling (0 for onion, 1 for apple)
        if "apple" in filename.lower():
            labels.append(1)
        elif "onion" in filename.lower():
            labels.append(0)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Red', 'Green', 'Blue'])
X = df
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to predict a new image
def predict_image(image_path):
    features = extract_features(image_path)
    new_data = pd.DataFrame([features], columns=['Red', 'Green', 'Blue'])
    prediction = model.predict(new_data)
    return "apple" if prediction[0] == 1 else "onion"

# Test with a new image
new_image = "C:\\Users\\rezafta\\Desktop\\ML\\ImageTag\\test_image.jpg"  # Replace with the path to your new image
result = predict_image(new_image)
print(f"This image is an {result}!")


new_image = "C:\\Users\\rezafta\\Desktop\\ML\\ImageTag\\test_image2.jpg"  # Replace with the path to your new image
result = predict_image(new_image)
print(f"This image is an {result}!")