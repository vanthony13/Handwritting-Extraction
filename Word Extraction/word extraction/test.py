from flask import Flask, request, render_template
import cv2
import pytesseract
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Read the file as an image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)


        # Preprocess the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Perform OCR on the preprocessed image
        text = pytesseract.image_to_string(gray)
        print(text)

        return render_template('index.html', text=text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
