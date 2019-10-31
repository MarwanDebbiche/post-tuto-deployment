import sys
import os
import random

from tqdm import tqdm

from flask import Blueprint, request, jsonify, Flask
import torch
import torch.nn.functional as F
import boto3

import db
import config
from ml.model import CharacterLevelCNN
from ml.utils import preprocess_input

app = Flask(__name__)
api = Blueprint('api', __name__)

# Load pytorch model for inference
model_path = './ml/checkpoints/model.pth'
model = CharacterLevelCNN()


def hook(t):
    def inner(bytes_amount):
        t.update(bytes_amount)
    return inner


if 'model.pth' not in os.listdir('./ml/checkpoints/'):
    print('downloading the trained model from s3')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('tuto-e2e-ml-trustpilot')
    file_object = s3.Object('tuto-e2e-ml-trustpilot', 'models/model.pth')
    filesize = file_object.content_length
    with tqdm(total=filesize, unit='B', unit_scale=True, desc='model.pth') as t:
        bucket.download_file('models/model.pth', model_path, Callback=hook(t))
else:
    print('model already saved to api/ml/checkpoints/model.pth')

if torch.cuda.is_available():
    trained_weights = torch.load(model_path)
else:
    trained_weights = torch.load(model_path, map_location='cpu')

model.load_state_dict(trained_weights)
model.eval()
print('PyTorch model loaded !')


@api.route('/predict', methods=['POST'])
def predict_rating():
    '''
    Endpoint to predict the rating using the
    review's text data.
    '''
    if request.method == 'POST':
        print('request.from =', request.form)
        if 'review' not in request.form:
            return jsonify({'error': 'no review in body'}), 400
        else:
            parameters = model.get_model_parameters()
            review = request.form['review']
            processed_input = preprocess_input(review, **parameters)
            prediction = model(processed_input)
            probabilities = F.softmax(prediction, dim=1)
            probabilities = probabilities.detach().cpu().numpy()
            output = probabilities[0][1]
            return jsonify(float(output))


@api.route('/review', methods=['POST'])
def post_review():
    '''
    Save review to database.
    '''
    if request.method == 'POST':
        expected_fields = [
            'review',
            'rating',
            'suggested_rating',
            'sentiment_score',
            'brand',
            'user_agent',
            'ip_address'
        ]
        if any(field not in request.form for field in expected_fields):
            return jsonify({'error': 'Missing field in body'}), 400

        query = db.Review.create(**request.form)

        return jsonify(query.serialize())


@api.route('/reviews', methods=['GET'])
def get_reviews():
    '''
    Get all reviews.
    '''
    if request.method == 'GET':
        query = db.Review.select()

        return jsonify([r.serialize() for r in query])


app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST)
