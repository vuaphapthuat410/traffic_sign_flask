from __future__ import division, print_function
from flask import Flask, request, jsonify
import utils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import os
from flask import Flask, request
# addition for this prj
import cv2
from efficientnet.tfkeras import EfficientNetB4
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sys


app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = './model/MyCheckPoint.h5'

# Load your trained model
model = load_model(MODEL_PATH)

@app.route('/')
def hello_world():
    return "Hello World!"

def model_predict(img_path, model):
    image = cv2.imread(img_path)
    x_input = cv2.resize(image, (32,32))
    x_input = cv2.cvtColor(x_input, cv2.COLOR_BGR2RGB)
    plt.imshow(x_input)
    x_input = x_input.reshape(1,32,32,3)
    x_input = x_input/255

    # Getting scores from forward pass of input image
    scores = model.predict(x_input)

    # Scores is given for image with 43 numbers of predictions for each class
    # Getting only one class with maximum value
    prediction = np.argmax(scores)
    print('ClassId:', prediction, file=sys.stdout)

    # Defining function for getting texts for every class - labels
    def label_text(file):
        # Defining list for saving label in order from 0 to 42
        label_list = []
        
        # Reading 'csv' file and getting image's labels
        r = pd.read_csv(file)
        # Going through all names
        for name in r['SignName']:
            # Adding from every row second column with name of the label
            label_list.append(name)
        
        # Returning resulted list with labels
        return label_list


    # Getting labels
    labels = label_text('./label_names.csv')

    # Printing label for classified Traffic Sign
    # print('Label:', labels[prediction], file=sys.stdout)
    # print('Total of softmax output:', scores, file=sys.stdout) 
    # print('Score:', np.max(scores), file=sys.stdout)
    return [labels[prediction], np.max(scores)]
    

@app.route('/predict', methods=['POST', 'GET'])
def classify():
    try:
        if request.method == 'POST':
            # check if the post request has the image part
            if 'image' not in request.files:
                return jsonify({
                    'message': 'No file part'
                }), 400
            file = request.files['image']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return jsonify({
                    'message': 'No selected file'
                }), 400
            filename = utils.save_upload_file(file)
            return jsonify({
                'method': "GET",
                'filename': filename
            })
        elif request.method == 'GET' and request.args.get('image_url', '') != '':
            image_url = request.args.get('image_url')
            filename = utils.download_image_from_url(image_url)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploaded', filename)
            predict = model_predict(file_path, model)

            print('Label:', predict[0], file=sys.stdout)
            print('Score:', predict[1], file=sys.stdout)

            str1 = str(predict[0]);
            str2 = str(predict[1]);

            # Arrange the correct return according to the model.
            return jsonify({
                'method': 'POST',
                'image_url': image_url,
                'file_name': filename,
                'label:': str1,
                'score' : str2
            })
        else:
            return jsonify({
                'message': 'Action is not defined!'
            }), 404
    except Exception as e:
        return repr(e), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)