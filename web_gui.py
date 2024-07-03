import os
import io
import imghdr
import base64
import logging
import settings
from waitress import serve
from matplotlib.figure import Figure
from paste.translogger import TransLogger
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, render_template, request
from detection_utils.universal_detection_api import UniversalObjectDetector
from detection_utils import detections_drawer as ddraw


def url_builder(url):
    return f"{settings.BASE_URL}{url}"

app = Flask(__name__)

@app.route(url_builder(""), methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", modelname=settings.MODEL_NAME)
    try:
        data = request.files["image"]
        if not data.filename:
            return render_template("index.html", error=f"missing image file!", modelname=settings.MODEL_NAME), 400
        if data.mimetype not in ["image/jpg", "image/jpeg", "image/png", "image/bmp"]:
            return render_template("index.html", error=f"invalid image type! {data.mimetype}", modelname=settings.MODEL_NAME), 400
        raw_data = data.stream.read()
        imgtype = imghdr.what(None, h=raw_data)
        if imgtype not in ["jpeg", "png", "bmp"]:
            return render_template("index.html", error=f"invalid image type! {data.mimetype}", modelname=settings.MODEL_NAME), 400
        detections = model.detect_objects(raw_data)
        fig = Figure()
        axes = fig.subplots()
        axes.axis("off")
        ddraw.box_drawer(raw_data, None, detections["boxes"], axes)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        payload = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return render_template("index.html", image=f"data:image/png;charset=utf-8;base64,{payload}",modelname=settings.MODEL_NAME)
    except Exception as e:
        app.logger.exception(e)
        return render_template("index.html", error="unable to process request.", modelname=settings.MODEL_NAME), 500

if __name__ == "__main__":
    global model
    logger = logging.getLogger("waitress")
    logger.setLevel(logging.INFO)
    app.logger.setLevel(logging.INFO)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
    app.logger.info(f"Loading {settings.MODEL_NAME} model.")
    model = UniversalObjectDetector(os.path.join("workspace", "exported_models", settings.MODEL_NAME), settings.CATEGORY_INDEX)
    app.logger.info(f"Loaded model")
    serve(
        TransLogger(app.wsgi_app),
        host=settings.HOST,
        port=settings.PORT,
        url_scheme="http",
        ident="Web server",
        max_request_body_size=10_485_760
    )
    # app.run(settings.HOST, settings.PORT, settings.DEBUG)
    