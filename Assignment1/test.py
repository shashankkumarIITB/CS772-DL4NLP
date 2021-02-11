from tensorflow.keras.models import load_model
from assign1 import preprocess_data

model = load_model('Assignment1.h5')

sentence = ["Showed up not how it's shown . Was someone's old toy. with paint on it.", 
            "It cannot be.."]

data = preprocess_data(sentence)
print(data)
