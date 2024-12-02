# Import necessary libraries
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the HAPPYDB dataset
df = pd.read_csv('export20112024_3.txt', sep='\t')

# Display the first few rows of the dataframe
print(df.head())

# Count the values in the predicted_category column
category_counts = df.predicted_category.value_counts()

# Visualize the category distribution
fig = go.Figure(
    data=[
        go.Bar(
            x=['achievement', 'affection', 'bonding', 'enjoy_the_moment', 
               'leisure', 'nature', 'exercise'],
            y=category_counts,
            text=category_counts,
            textposition='auto',
        )
    ]
)

# Show the bar chart
fig.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_hm'], df['predicted_category'], test_size=0.2, random_state=7
)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data; transform the test data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Encode the target variable (categories)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize and train an SVM model
model_svm = SVC(kernel='linear')
model_svm.fit(tfidf_train, y_train_encoded)

# Evaluate the SVM model on training data
y_train_pred = model_svm.predict(tfidf_train)
train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
print(f"Training accuracy of SVM: {train_accuracy:.2f}")

# Evaluate the SVM model on test data
y_test_pred = model_svm.predict(tfidf_test)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
print(f"Testing accuracy of SVM: {test_accuracy:.2f}")

# Define a function to predict the category of happiness
def predict_happiness_reason(example):
    """
    Predict the category of happiness based on the input text.
    
    Args:
        example (str): Input text describing a happy moment.
    
    Returns:
        str: Predicted category of happiness.
    """
    example_tfidf = tfidf_vectorizer.transform([example])
    predicted_class_encoded = model_svm.predict(example_tfidf)
    predicted_class = label_encoder.inverse_transform(predicted_class_encoded)
    return predicted_class[0]

# Examples of predictions
examples = [
    "15 days ago I got drunk for the very first time and felt very awesome. "
    "I drank at my friend's birthday party and had a hard party time. "
    "It was really a happy moment for me that I won't forget at all.",

    "I always enjoy the opportunity to sit by the fire and sing my favorite songs on hikes.",

    "My grandmother, whom I hadn't seen for 2 years, came to visit us."
]

for example in examples:
    predicted_category = predict_happiness_reason(example)
    print(f"The model predicts the reason for happiness as: {predicted_category}")
