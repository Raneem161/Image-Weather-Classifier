from fastapi import FastAPI, File, UploadFile
from PIL import Image
from keras.models import load_model
import numpy as np

# Loading the trained model
model = load_model('my_model')

app = FastAPI()

# Defining the class names 
class_names = ['dew',
 'fogsmog',
 'frost',
 'glaze',
 'hail',
 'lightning',
 'rain',
 'rainbow',
 'rime',
 'sandstorm',
 'snow'] 

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg', 'image/png']:
        return {"error": "Only JPG and PNG images are supported."}
    
    # Reading and preprocessing the image
    image = Image.open(file.file)
    image = image.resize((256, 256))
    img_array = np.array(image)
    img_array = img_array / 255.0  # Rescaling the image

    # Predicting the class
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
