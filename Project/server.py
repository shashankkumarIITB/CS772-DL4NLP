import os, re, string
from flask import Flask
from flask import request, render_template
from gtts import gTTS
import pytesseract
from PIL import Image
from tensorflow.keras.models import load_model
from google_trans_new import google_translator  
# from translate import Translator

# Import prediction functions
from models.attention.predict import predict as predict_attention
from models.encoder.predict import predict as predict_encoder

# Set the following environment vairables
# $env:FLASK_APP = "server.py"
# $env:FLASK_DEBUG = 1
# To start the server, use - flask run

# Disable tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'D:/Apps/Tesseract-OCR/tesseract.exe'

app = Flask(__name__)
IMAGE_UPLOAD_DIRECTORY = os.path.join('static', 'image')
AUDIO_TTS_DIRECTORY = os.path.join('static', 'audio')
# TRANSLATOR = Translator(to_lang='hi')
TRANSLATOR = google_translator()  

# Function to get the text from the image after preprocessing
def getTextFromImage(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    # Preprocess the text
    text = "".join([character for character in text if character not in string.punctuation])
    text = re.sub('\s+', ' ', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def translate():
    # Title of the page
    title='CS772 Project | OCR-MT-TTS'

    if request.method == 'POST':
        # Check if image is uploaded
        image_present = request.form['image_present']
        if image_present == 'true':
            # Save the uploaded image
            ocr_image = request.files['ocr_image']
            image_extension = ocr_image.filename.rsplit('.', 1)[1].lower()
            image_path = os.path.join(IMAGE_UPLOAD_DIRECTORY, f'image.{image_extension}')
            ocr_image.save(image_path)

            # Get the text from the image
            text_extracted = getTextFromImage(image_path)
        else:
            # Directly get the text from the user
            text_extracted = request.form['text_to_translate']

        # Expected translation
        # expected_tranlation = TRANSLATOR.translate(text_extracted)
        expected_translation = TRANSLATOR.translate(text_extracted, lang_tgt='hi')  
        # expected_translation = 'Enable during demo, Line 64'

        # The architecure selected for machine translation
        architecture = request.form['architecture']
        # if architecture == 'transformer':
        #     text_translated = 'Hello World'
        if architecture ==  'attention':
            text_translated = predict_attention([text_extracted])
        elif architecture == 'encoder':
            text_translated = predict_encoder([text_extracted])

        # Generate the speech from the translated text
        tts = gTTS(text_translated)
        audio_path = os.path.join(AUDIO_TTS_DIRECTORY, 'audio.mp3')
        tts.save(audio_path)

        # Context dictionary for the template
        context = {
            'title': title,
            'text_extracted': text_extracted,
            'text_translated': text_translated,
            'expected_translation' : expected_translation,
            'audio_path': audio_path
        }

        if image_present == 'true':
            context['image_present'] = True
            context['image_path'] = image_path
        
        # Render the translated text along with the uploaded image
        return render_template('translate.html', **context)
    return render_template('home.html', title=title)

@app.route('/test', methods=['GET'])
def test():
    extension = 'jpg'
    image_path=os.path.join(IMAGE_UPLOAD_DIRECTORY, f'image3.{extension}')
    return render_template('test.html', image_path=image_path)

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True, threaded=True,port=5000)
