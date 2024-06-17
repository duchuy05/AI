import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

MODEL_PATH = "traffic_sign_model_final.h5"
model = load_model(MODEL_PATH)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'Duong Cam'
    elif classNo == 1:
        return 'Cam di nguoc chieu'
    elif classNo == 2:
        return 'Cam o to'
    elif classNo == 3:
        return 'Cam o to re phai'
    elif classNo == 4:
        return 'Cam o to re trai'
    elif classNo == 5:
        return 'Cam xe may'
    elif classNo == 6:
        return 'Cam xe tai'
    elif classNo == 7:
        return 'Cam o to khach va o to tai'
    elif classNo == 8:
        return 'Cam xe dap'
    elif classNo == 9:
        return 'Cam nguoi di bo'
    elif classNo == 10:
        return 'Han che chieu cao xe'
    elif classNo == 11:
        return 'Han che chieu rong xe'
    elif classNo == 12:
        return 'Dung lai'
    elif classNo == 13:
        return 'Cam re trai'
    elif classNo == 14:
        return 'Cam re phai'
    elif classNo == 15:
        return 'Cam quay dau'
    elif classNo == 16:
        return 'Cam o to quay dau'
    elif classNo == 17:
        return 'Toc do toi da'
    elif classNo == 18:
        return 'Cam bop coi'
    elif classNo == 19:
        return 'Cam dung va do xe'
    elif classNo == 20:
        return 'Cam do xe'
    elif classNo == 21:
        return 'Giao nhau voi duong khong uu tien'
    elif classNo == 22:
        return 'Giao nhau voi duong khong uu tien'
    elif classNo == 23:
        return 'Giao nhau voi duong uu tien'
    elif classNo == 24:
        return 'Giao nhau voi tin hieu den'
    elif classNo == 25:
        return 'Giao nhau voi duong sat co rao chan'
    elif classNo == 26:
        return 'Giao nhau voi duong sat khong rao chan'
    elif classNo == 27:
        return 'Duong khong bang phang'
    elif classNo == 28:
        return 'Nguoi di bo cat ngang'
    elif classNo == 29:
        return 'Nguy hiem tre em qua duong'
    elif classNo == 30:
        return 'Cong truong'
    elif classNo == 31:
        return 'Duong sat cat duong bo'
    elif classNo == 32:
        return 'Di cham'
    elif classNo == 33:
        return 'Noi giao nhau chay theo vong xuyen'
    elif classNo == 34:
        return 'Duong danh cho nguoi di bo'
    elif classNo == 35:
        return 'Toc do toi thieu cho phep'
    elif classNo == 36:
        return 'Het han che toc do toi thieu'
    elif classNo == 37:
        return 'Tuyen duong cau vuot bat qua'
    elif classNo == 38:
        return 'Cac xe chi duoc re trai hoac re phai'
    elif classNo == 39:
        return 'Huong di vong chuong ngai vat sang trai'
    else:
        return 'Unknown class number'

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(32, 32))
    img = np.asarray(img)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)  # Reshape for model input
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    probabilityValue = np.amax(predictions)
    preds = getClassName(classIndex)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == "__main__":
    app.run(debug=True)
