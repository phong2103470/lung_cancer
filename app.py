from flask import Flask, request, render_template, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Thư mục lưu ảnh tải lên

# Danh sách các class
class_names = ['adenocarcinoma - Ung thư biểu mô tuyến', 'large.cell.carcinoma - Ung thư biểu mô tế bào lớn', 'normal - Bình thường', 'squamous.cell.carcinom - Ung thư biểu mô tế bào vảy']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    selected_model = None  # Giữ lại model đã chọn

    if request.method == 'POST':
        if 'image' not in request.files or 'model' not in request.form:
            return "Vui lòng chọn mô hình và tải lên ảnh.", 400

        file = request.files['image']
        selected_model = request.form['model']  # Lưu model đã chọn

        if file.filename == '':
            return "Chưa chọn file. Vui lòng chọn một file và thử lại.", 400

        # Lưu file vào thư mục uploads
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Load mô hình được chọn
        try:
            model = tf.keras.models.load_model(selected_model)
            
            # Xử lý ảnh và dự đoán
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            prediction = class_names[np.argmax(preds)]
        
        except Exception as e:
            return f"Lỗi xử lý ảnh hoặc mô hình: {str(e)}", 500

    return render_template('index.html', prediction=prediction, img_path=img_path, selected_model=selected_model)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)