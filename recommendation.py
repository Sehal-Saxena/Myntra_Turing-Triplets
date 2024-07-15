import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import json
import warnings

warnings.filterwarnings('ignore')

# Function to preprocess images
def preprocess_images(img_paths, target_size=(224, 224)):
    img_arrays = []
    for img_path in img_paths:
        full_path = os.path.join('images', img_path)
        img = Image.open(full_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_arrays.append(img_array)
    img_arrays = np.array(img_arrays)
    img_arrays = preprocess_input(img_arrays)
    return img_arrays

# Function to extract image features
def extract_image_features(vgg16_model, img_arrays):
    features = vgg16_model.predict(img_arrays, batch_size=32)  # Use batch processing for faster prediction
    features_flat = features.reshape(features.shape[0], -1)
    return features_flat

# Function to generate comparison report
def generate_comparison_report(user_details, data, model, encoder, scaler, vgg16_model):
    img_array = preprocess_images([user_details['image_path']], target_size=(224, 224))
    img_features = extract_image_features(vgg16_model, img_array).reshape(1, -1)
    
    details_df = pd.DataFrame([user_details])
    
    # Encode the details
    for col in ['weather', 'Occasion', 'material']:
        if details_df[col].iloc[0] not in encoder.categories_[0]:
            details_df[col] = f'unknown_{col}'
    
    categorical_features = encoder.transform(details_df[['weather', 'Occasion', 'material']])
    
    user_features = np.hstack((img_features, categorical_features.toarray()))
    user_features = scaler.transform(user_features)
    
    prediction = model.predict(user_features)
    
    comparison_report = pd.DataFrame({
        'Clothing Item': [user_details['image_path']],
        'Suitability': ['fit' if prediction[0] == 1 else 'not fit']
    })
    
    matched_categories = []
    not_matched_categories = []
    
    for col, category in zip(['weather', 'Occasion', 'material'], encoder.categories_):
        user_value = user_details[col]
        if user_value in category:
            category_index = np.where(category == user_value)[0][0]
            if categorical_features[0, category_index] == 1:
                matched_categories.append(col)
            else:
                not_matched_categories.append(col)
        else:
            not_matched_categories.append(col)
    
    if matched_categories:
        comparison_report['Matched Categories'] = ', '.join(matched_categories)
    if not_matched_categories:
        comparison_report['Not Matched Categories'] = ', '.join(not_matched_categories)
    
    return comparison_report

def main():
    # Load dataset
    data = pd.read_csv('data.csv')

    # Initialize VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # Preprocess images and extract features
    img_paths = data['filename'].tolist()
    img_arrays = preprocess_images(img_paths)
    image_features = extract_image_features(vgg16_model, img_arrays)

    # Add image features to the dataset
    data['image_features'] = list(image_features)

    # Encoding categorical data
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[['weather', 'Occasion', 'material']])

    # Combine features
    combined_features = np.hstack((image_features, encoded_features.toarray()))

    # Encode labels (all labels will be 'fit' since the dataset only contains fitting clothes)
    data['encoded_labels'] = 1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(combined_features, data['encoded_labels'], test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Load user details from data.json based on intent
    with open('data.json', 'r') as f:
        user_data = json.load(f)

    # Manually entered clothes details
    clothes_details = [
        {
            'weather': 'Sunny',
            'Occasion': 'Casual',
            'material': 'Cotton',
            'image_path': '20240711223254928829.jpg'  # Replace with your actual image path
        },
        {
            'weather': 'Rainy',
            'Occasion': 'Formal',
            'material': 'Wool',
            'image_path': '20240711223300717155.jpg'  # Replace with your actual image path
        }
        ,
          {
            'weather': 'Sunny',
            'Occasion': 'haldi',
            'material': 'cotton',
            'image_path': '20240711223256727223.jpg'  # Replace with your actual image path
        }
        # Add more items as needed
    ]

    # Process each manually entered clothes detail and generate comparison report
    reports = []
    for clothes_detail in clothes_details:
        report = generate_comparison_report(clothes_detail, data, model, encoder, scaler, vgg16_model)
        reports.append(report)

    # Convert the list of DataFrames to a single DataFrame
    comparison_reports = pd.concat(reports).reset_index(drop=True)

    # Ensure 'Matched Categories' column exists
    if 'Matched Categories' in comparison_reports.columns:
        # Score based on the number of matched categories
        comparison_reports['Score'] = comparison_reports.apply(
            lambda row: row['Matched Categories'].count(',') + 1 if pd.notna(row['Matched Categories']) else 0, axis=1)
    else:
        # Handle case where 'Matched Categories' column is absent
        comparison_reports['Score'] = 0

    # Sort by score in descending order
    sorted_reports = comparison_reports.sort_values(by='Score', ascending=False)

    # Display the most suitable clothes or the closest match
    suitable_clothes = sorted_reports[sorted_reports['Suitability'] == 'fit']
    if not suitable_clothes.empty:
        print("Most suitable clothes:")
        print(suitable_clothes)
    else:
        print("No exact match found. Closest match:")
        print(sorted_reports.head(1))

if __name__ == "__main__":
    main()
