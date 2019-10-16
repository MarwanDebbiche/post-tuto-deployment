from flask import Blueprint, request, jsonify, Flask
import random
import db

api = Blueprint('api', 'api', url_prefix='/api')


@api.route('/predict-rating', methods=['POST'])
def predict_rating():
    '''
    Endpoint to predict the rating using the
    review's text data.
    '''
    if request.method == 'POST':
        if 'review' not in request.form:
            return jsonify({'error': 'no review in body'}), 400

        return jsonify(random.randint(1, 5))


@api.route('/review', methods=['POST'])
def post_review():
    '''
    Save review to database.
    '''
    if request.method == 'POST':
        if any(field not in request.form for field in ['review', 'rating', 'rating']):
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


app = Flask(__name__)
app.register_blueprint(api)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
