import os
import json
from flask import Flask, flash, jsonify, request, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret key"

path = os.getcwd()
upload_folder = os.path.join(path,'images')

allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.isdir(upload_folder):
    os.mkdir(upload_folder)

app.config['UPLOAD_FOLDER'] = upload_folder

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route("/api/upload",methods=['GET','POST'])
@cross_origin()
def upload_images():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))

    return 

if __name__ == "__main__":
    app.run(host="0.0.0.0",port = 5000)