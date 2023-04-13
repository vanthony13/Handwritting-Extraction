from google.cloud import vision
import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Vitoria\Desktop\TFC\mltu-main (copy)\TFC_OCR_Key.json'

def extract_handwritten_text(image_path, mode='word'):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    if mode == 'word':
        response = client.document_text_detection(image=image)
        text = ''
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            text += symbol.text
                        text += ' '
        return text.strip()
    elif mode == 'sentence':
        response = client.document_text_detection(image=image)
        text = ''
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            text += symbol.text
                        text += ' '
                    text = text.strip()
                    if len(text) <= 200:
                        text += ' '
                    else:
                        return 'Ups... Limit Exceeded'
        return text.strip()
