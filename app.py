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

# Response model
response_model = api.model('Response', {
    'detected_birds': fields.List(fields.String, description='The identified birds'),
})


@ns.route('/identify')
@api.expect(upload_parser)
class BirdIdentification(Resource):
    @api.response(200, 'Success', response_model)
    @api.response(400, 'Validation Error')
    @api.doc(description="Identify birds in an uploaded image.")
    def post(self):
        args = upload_parser.parse_args()
        image = args['image']
        result = main.process_image(image)
        return {'detected_birds': result}


if __name__ == '__main__':
    app.run(debug=True)
