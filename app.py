import pickle
import re
import nltk
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Load the trained models and vectorizer
with open('model_logistic_regression.pkl', 'rb') as f:
    model_logreg = pickle.load(f)

with open('model_naive_bayes.pkl', 'rb') as f:
    model_nb = pickle.load(f)

with open('model_svc.pkl', 'rb') as f:
    model_svc = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

# Initialize stopwords and stemmer
stop_words = stopwords.words('english')
ps = PorterStemmer()

# Preprocess function to clean the review text
def preprocess_review(review):
    review = re.sub("[^a-zA-Z | ^\w+'t]", ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    return review

# Convert prediction from numeric (0, 1) to string ('negative', 'positive')
def convert_prediction(prediction):
    return 'positive' if prediction == 1 else 'negative'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the form
    review = request.form['review']
    
    # Preprocess the review
    review = preprocess_review(review)
    
    # Convert the review to a feature vector
    review_vector = cv.transform([review]).toarray()
    
    # Get predictions from all models
    logreg_pred = model_logreg.predict(review_vector)[0]
    nb_pred = model_nb.predict(review_vector)[0]
    svc_pred = model_svc.predict(review_vector)[0]

    # Convert predictions to 'positive' or 'negative'
    logreg_pred = convert_prediction(logreg_pred)
    nb_pred = convert_prediction(nb_pred)
    svc_pred = convert_prediction(svc_pred)

    # Return the prediction results
    return render_template('index.html', review=review, logreg_pred=logreg_pred, nb_pred=nb_pred, svc_pred=svc_pred)

if __name__ == '__main__':
    app.run(debug=True)
