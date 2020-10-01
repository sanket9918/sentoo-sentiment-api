import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from flask_cors import CORS
from flask import Flask, request, jsonify, make_response
from keras.models import load_model
import sys
from flask_restx import Api, Resource, fields

use = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

flask_app = Flask(__name__)
CORS(flask_app)
app = Api(app=flask_app,
          version="1.0",
          title="Sentiment Analysis API",
          description="Predict results using a trained model")


name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params',
                  {'comment': fields.String(required=True,
                                            description="Comment for the particular property",
                                            help="Comment cannot be left blank"),


                   })

filename = 'my-model.h5'
predictor = load_model(filename)


@name_space.route("/")
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)
    def post(self):
        try:
            formData = request.json
            input_raw = [val for val in formData.values()]
            input_final = []
            emb = use(input_raw[0])
            review_emb = tf.reshape(emb, [-1]).numpy()
            input_final.append(review_emb)
            input_final = np.array(input_final)
            prediction = predictor.predict(
                input_final[:1])
            pred_comment = ""
            comment_code = 0
            if(np.argmax(prediction) == 0):
                pred_comment = "Bad"
                comment_code = 0
            else:
                pred_comment = "Good"
                comment_code = 1

            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": "The predicted rating of the property is " + pred_comment,
                "comment_code": comment_code
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "result": "Please review your response and try again",
                "error": str(error)
            })
