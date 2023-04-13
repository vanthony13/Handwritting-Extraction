import onnxruntime
import numpy as np
import cv2
import yaml

def extract_handwritten_text(image_path, mode='word'):
    if mode == 'word':
        model_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301111911\model.onnx'
        configs_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301111911\configs.yaml'
    elif mode == 'sentence':
        model_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301131202\model.onnx'
        configs_path = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\Word Extraction\word extraction\Models\03_handwriting_recognition\202301131202\configs.yaml'

    with open(configs_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    input_height = configs['height']
    input_width = configs['width']
    vocab = configs['vocab']

    sess = onnxruntime.InferenceSession(model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (input_width, input_height))
    image = np.array(image, dtype=np.float32)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)  # Add the channel dimension
    image = np.expand_dims(image, axis=0)  # Add the batch dimension

    preds = sess.run(None, {'input': image})
    decoded_preds = decode_predictions(preds, vocab)

    return decoded_preds[0]

def decode_predictions(preds, vocab):
    preds = np.argmax(preds, axis=-1)
    decoded_preds = []
    for pred in preds:
        decoded_pred = []
        for idx in pred:
            if idx == len(vocab):
                break
            decoded_pred.append(vocab[idx])
        decoded_preds.append(''.join(decoded_pred))
    return decoded_preds
