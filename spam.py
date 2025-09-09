# Install necessary packages if not already installed
# !pip install pandas scikit-learn nltk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords
nltk.download('stopwords')

# dataset will be loaded

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Preprocess the text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_message'] = df['message'].apply(preprocess_text)

# Convert the text to numeric features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])

# Labels: spam=1, ham=0
y = df['label'].map({'ham': 0, 'spam': 1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Traininig the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# performance evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")



def predict_message(text):
    # Preprocess the text (same as training)
    text_clean = preprocess_text(text)
    # Convert to TF-IDF features
    text_vec = vectorizer.transform([text_clean])
    # Predict label
    pred = model.predict(text_vec)[0]
    # Map prediction to label
    label = 'spam' if pred == 1 else 'ham'
    return label

# Test with the new messages
test_messages = [
    "Congratulations! You've won a $100000 Lottery. Go to http://bit.ly/123456 to claim now.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Reply with your password immediately."
]

for msg in test_messages:
    print(f"Message: {msg}")
    print(f"Prediction: {predict_message(msg)}\n")
