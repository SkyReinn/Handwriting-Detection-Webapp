#imports
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from PIL import Image

import easyocr
import TextSequencing

import keras_ocr
import matplotlib.pyplot as plt



def searchFile():
    # This is to get the directory that the program
    # is currently running in.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    success = False
    for root, dirs, files in os.walk("static/files"):
        for file in files:
            # change the extension from '.mp3' to
            # the one of your choice.
            if file.endswith('.jpg') or file.endswith('png') or file.endswith("JPG"):
                return (root+'/'+str(file))
                
        
def analyze(file):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Load the scanned image
    image = file

    # Perform OCR on the image
    predictions = reader.readtext(image, paragraph=True,allowlist='.!,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    result = ""

    # Turn the result into a single list
    for prediction in predictions:
        result += prediction[1]
        result += " "

    result = TextSequencing.Normalize(result)
    return result

app = Flask(__name__,template_folder="templates")
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():

        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))

        path = searchFile()
        #Image.open(file).show()
        result =analyze(path) #get returned words
        os.remove(path) #remove file from path after having it added from front end 
        stats="Detected Text: " + ' '.join([str(elem) for elem in result])
        return render_template('index.html', form=form,stats=stats) #change with web page to change to 
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

