import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
import albumentations as A

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        # If the input image has only 1 channel, convert it to 3 channels
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


def preprocess_image(image: np.ndarray):
    # Resize the image to a larger size
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce the noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Apply adaptive thresholding to better adapt to variations in lighting and contrast
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply image dilation to thicken the characters
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)

    # Use a mask to remove the background and isolate the characters
    mask = np.zeros_like(img_dilated)
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    img_masked = cv2.bitwise_and(image, image, mask=mask)

    return img_masked


if __name__ == "__main__":
    model1_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301111911\model.onnx'
    configs1_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301111911\configs.yaml'
    model1_configs = BaseModelConfigs.load(configs1_path)
    model1 = ImageToWordModel(model_path=model1_path, char_list=model1_configs.vocab)

    model2_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301131202\model.onnx'
    configs2_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301131202\configs.yaml'
    model2_configs = BaseModelConfigs.load(configs2_path)
    model2 = ImageToWordModel(model_path=model2_path, char_list=model2_configs.vocab)

    image_path = r'C:\Users\Vitoria\Desktop\test\input6.jpg'
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)

    # Prompt user for type of prediction
    while True:
        prediction_type = input("Enter 'word' or 'sentence' to predict: ")
        if prediction_type.lower() == 'word':
            prediction_model = model1
            break
        elif prediction_type.lower() == 'sentence':
            prediction_model = model2
            break
        else:
            print("Invalid input, please enter 'word' or 'sentence'.")

    # Apply image augmentation
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.Sharpen(p=0.5)
    ])
    augmented_image = transform(image=image)['image']

    try:
        # Make prediction using selected model
        prediction_text = prediction_model.predict(preprocessed_image)
        print(f"Prediction: {prediction_text}")

    except Exception as e:
        print(f"Error: {e}")
