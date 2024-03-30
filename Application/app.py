from flask import Flask, request, render_template, session, redirect
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


app = Flask(__name__)
app.secret_key = b'v43gr8w46uh'


@app.route('/')
def home():  # put application's code here
    return render_template("upload.html")


@app.route('/recognize', methods=['POST'])
def recognize():  # put application's code here
    picture = request.files.get("picture")
    if not picture:
        return render_template("upload.html", noFile=1)
    src = r"\static\uploads\test.jpg"
    path = os.path.abspath(os.path.dirname(__file__)) + src
    picture.save(path)
    print(path)
    images_array = np.array([path])
    labels_array = np.array([""])
    df_test = pd.DataFrame({'image': images_array, 'label': labels_array})
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col='image',
        y_col='label',
        target_size=(50, 50),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    model = load_model('plant-disease-resnet.h5')
    predictions = model.predict(test_generator)
    lables = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    result = lables[np.argmax(predictions[0])].split("___", 1)
    return render_template("result.html", picture_src=src, result=result)


if __name__ == '__main__':
    app.run()
