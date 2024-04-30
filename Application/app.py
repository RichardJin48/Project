from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


app = Flask(__name__)
app.secret_key = b'v43gr8w46uh'

model = load_model('plant-disease-resnet.h5')


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
    predictions = model.predict(test_generator)
    labels = ['Apple___Apple Scab', 'Apple___Black Rot', 'Apple___Cedar Apple Rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry (Including Sour)___Powdery Mildew', 'Cherry (Including Sour)___healthy',
               'Corn (Maize)___Cercospora Leaf Spot/Gray Leaf Spot', 'Corn (Maize)___Common Rust',
               'Corn (Maize)___Northern Leaf Blight', 'Corn (Maize)___healthy', 'Grape___Black Rot',
               'Grape___Esca (Black Measles)', 'Grape___Leaf Blight (Isariopsis Leaf Spot)', 'Grape___healthy',
               'Orange___Haunglongbing (Citrus Greening)', 'Peach___Bacterial Spot', 'Peach___healthy',
               'Pepper (Bell)___Bacterial Spot', 'Pepper (Bell)___healthy', 'Potato___Early Blight',
               'Potato___Late Blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery Mildew', 'Strawberry___Leaf Scorch', 'Strawberry___healthy', 'Tomato___Bacterial Spot',
               'Tomato___Early Blight', 'Tomato___Late Blight', 'Tomato___Leaf Mold', 'Tomato___Septoria Leaf Spot',
               'Tomato___Spider Mites/Two-spotted Spider Mite', 'Tomato___Target Spot',
               'Tomato___Tomato Yellow Leaf Curl Virus', 'Tomato___Tomato Mosaic Virus', 'Tomato___healthy']
    result = labels[np.argmax(predictions[0])].split("___", 1)
    return render_template("result.html", picture_src=src, result=result)


if __name__ == '__main__':
    app.run()
