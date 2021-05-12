# from gtts import gTTS
# tts = gTTS('जोखिम लो या मौका गवां दो')
# tts.save('translate.mp3')

import re, string
text = 'Sunny Deol drives on a sunny day'
text = "".join([character for character in text if character not in string.punctuation])
text = re.sub('\s+', ' ', text)
print(text)

from translate import Translator
translator = Translator(to_lang='hi')
translation = translator.translate(text)
print(translation)

# import pytesseract
# from PIL import Image
# pytesseract.pytesseract.tesseract_cmd = 'D:/Apps/Tesseract-OCR/tesseract.exe'
# print(pytesseract.image_to_string(Image.open('./image/image.png'), lang='eng'))