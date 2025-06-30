import os
import cv2
import numpy as np
import traceback
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
from polarini_focusini import detect_infocus_mask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Проверка расширения файла
            if not allowed_file(file.filename):
                return render_template('index.html', error="Неподдерживаемый формат файла. Используйте JPG или PNG.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('\\', '/'))
            file.save(upload_path)

            img = cv2.imread(upload_path)
            if img is None:
                return render_template('index.html', error="Ошибка чтения изображения. Попробуйте другой файл.")

            max_size = 1024
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            # Обработка изображения
            mask = detect_infocus_mask(
                img,
                limit_with_circles_around_focus_points=True,
                ignore_cuda=True,  # На PythonAnywhere нет GPU
                verbose=True
            )

            # Сохранение результатов
            mask_filename = f"mask_{filename}"
            mask_path = os.path.join(app.config['RESULT_FOLDER'], mask_filename.replace('\\', '/'))
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)

            # Создание визуализации
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            result_img = np.where(mask[:, :, np.newaxis], img, img_gray)

            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename.replace('\\', '/'))
            cv2.imwrite(result_path, result_img)

            return redirect(url_for('show_result',
                                    original=filename,
                                    mask=mask_filename,
                                    result=result_filename))

    except Exception as e:
        app.logger.error(f"Ошибка обработки: {str(e)}")
        app.logger.error(traceback.format_exc())
        return render_template('index.html', error=f"Ошибка обработки: {str(e)}")

    return redirect(request.url)


@app.route('/result')
def show_result():
    try:
        original = request.args.get('original', '')
        mask = request.args.get('mask', '')
        result = request.args.get('result', '')

        # Проверка существования файлов
        if not all([
            os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], original)),
            os.path.exists(os.path.join(app.config['RESULT_FOLDER'], mask)),
            os.path.exists(os.path.join(app.config['RESULT_FOLDER'], result))
        ]):
            return render_template('index.html', error="Результаты обработки не найдены. Попробуйте загрузить снова.")

        return render_template('result.html',
                               original=os.path.join('uploads', original).replace('\\', '/'),
                               mask=os.path.join('results', mask).replace('\\', '/'),
                               result=os.path.join('results', result).replace('\\', '/'))

    except Exception as e:
        app.logger.error(f"Ошибка отображения результатов: {str(e)}")
        return render_template('index.html', error="Ошибка отображения результатов")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


def secure_filename(filename):
    # Простая очистка имени файла
    keepchars = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)