import cv2
import numpy as np
import tensorflow as tf
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from model import train_model
from configs import ModelConfigs

# Load the trained OCR model
configs = ModelConfigs()
model = train_model(
    input_dim=(configs.height, configs.width, 1),
    output_dim=len(configs.vocab)
)
model.load_weights(configs.model_path)

# Define data preprocessing pipeline
preprocessors = [
    ImageReader(),
    ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
]

transformers = [
    LabelIndexer(configs.vocab),
    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
]

# Define function to apply OCR on an image and return extracted text
def extract_text(image_path):
    # Load the image file specified by the user
    image = cv2.imread(image_path)

    # Preprocess the image
    for preprocessor in preprocessors:
        image = preprocessor(image)

    image = np.expand_dims(image, axis=-1)

    # Apply OCR on the preprocessed image using the loaded model
    prediction = model.predict(np.array([image]))
    decoded_text = tf.keras.backend.ctc_decode(prediction, input_length=[prediction.shape[1]], greedy=True)[0][0]
    extracted_text = tf.strings.reduce_join(configs.label_to_text(decoded_text)).numpy().decode('utf-8')

    return extracted_text
