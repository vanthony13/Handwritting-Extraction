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
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":
    model1_path = r'C:\Users\Vitoria\Desktop\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301111911\model.onnx'
    configs1_path = r'C:\Users\Vitoria\Desktop\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301111911\configs.yaml'
    model1_configs = BaseModelConfigs.load(configs1_path)
    model1 = ImageToWordModel(model_path=model1_path, char_list=model1_configs.vocab)

    model2_path = r'C:\Users\Vitoria\Desktop\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301131202\model.onnx'
    configs2_path = r'C:\Users\Vitoria\Desktop\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301131202\configs.yaml'
    model2_configs = BaseModelConfigs.load(configs2_path)
    model2 = ImageToWordModel(model_path=model2_path, char_list=model2_configs.vocab)

    image_path = r'C:\Users\Vitoria\Desktop\mltu-main (copy)\Word Extraction\word extraction\Datasets\input13.jpg'
    image = cv2.imread(image_path)

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
        prediction_text = prediction_model.predict(augmented_image)
        print(f"Prediction: {prediction_text}")

    except Exception as e:
        print(f"Error: {e}")
