from flask import Flask, jsonify
from flask import request
import random


app = Flask(__name__)


@app.route('/predict-rating', methods=['POST'])
def predict_rating():
    '''
    Endpoint to predict the rating using the
    review's text data.
    '''
    if request.method == 'POST':
        if 'review' not in request.form:
            return jsonify({'error': 'no sting in body'}), 400

        return jsonify(random.randint(1, 5))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
