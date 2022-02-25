from flask import Flask, render_template, request
import numpy as np     # for mathematic equation
from nltk.corpus import stopwords   # to get collection of stopwords
from sklearn.model_selection import train_test_split       # for splitting dataset
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.layers import Embedding, LSTM, Dense # layers of the architecture
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from tensorflow.keras.models import load_model   # load saved model
import re
import pickle
import nltk
nltk.download('stopwords')
app = Flask(__name__)


loaded_model = load_model('models/LSTM.h5')
token = pickle.load(open('models/token.pkl','rb'))
english_stops = set(stopwords.words('english'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        review = request.form['Reviews']
        regex = re.compile(r'[^a-zA-Z\s]')
        review = regex.sub('', review)
        print('Cleaned: ', review)

        words = review.split(' ')
        filtered = [w for w in words if w not in english_stops]
        filtered = ' '.join(filtered)
        filtered = [filtered.lower()]
        max_length = 131
        tokenize_words = token.texts_to_sequences(filtered)
        tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
        result = loaded_model.predict(tokenize_words)
        if result >= 0.7:
            print('positive')  
            my_prediction =  1
        else:
            print('negative')
            my_prediction =  0
        
    return render_template('result.html',prediction = my_prediction)




if __name__ == '__main__':
    app.run(debug=True)