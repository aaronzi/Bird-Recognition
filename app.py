from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
import werkzeug
import main

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Bird Recognition API', description='A simple API for bird detection and identification')

ns = api.namespace('birds', description='Bird operations')

# Define the expected input model for documentation
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files', required=True, help='Image file')

# Response model for image containing multiple birds
response_model_multi = api.model('Response_Multi', {
    'detected_birds': fields.List(fields.String, description='The identified birds'),
})

# Response model for image containing a single bird
response_model_single = api.model('Response_Single', {
    'detected_bird': fields.String(description='The identified bird'),
})

@ns.route('/identify-multi')
@api.expect(upload_parser)
class BirdIdentificationMulti(Resource):
    @api.response(200, 'Success', response_model_multi)
    @api.response(400, 'Validation Error')
    @api.doc(description="Identify birds in an uploaded image. This first detects birds in the image and then identifies each detected bird.")
    def post(self):
        args = upload_parser.parse_args()
        image = args['image']
        result = main.process_image_multi(image)
        return {'detected_birds': result}
    

@ns.route('/identify-single')
@api.expect(upload_parser)
class BirdIdentificationSingle(Resource):
    @api.response(200, 'Success', response_model_single)
    @api.response(400, 'Validation Error')
    @api.doc(description="Identify a single bird in an uploaded image. This takes the uploaded image and identifies the bird in the image.")
    def post(self):
        args = upload_parser.parse_args()
        image = args['image']
        result = main.process_image_single(image)
        return {'detected_bird': result}


if __name__ == '__main__':
    app.run(debug=True)
