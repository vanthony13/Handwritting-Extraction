from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from extract_text import extract_handwritten_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        mode = request.form['mode']  # Update this line
        image_name = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        image.save(image_path)
        extracted_text = extract_handwritten_text(image_path, mode)

        if mode == 'sentence' and len(extracted_text.split()) > 1:
            extracted_text = extracted_text[:200]
        elif mode == 'word' and len(extracted_text.split()) > 1:
            extracted_text = 'Ups... Word only'

        return render_template('index.html', extracted_text=extracted_text)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
