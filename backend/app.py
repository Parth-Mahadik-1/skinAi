from flask import Flask, render_template, request, redirect, url_for,session,send_file
#from tensorflow import keras 
#import tensorflow as tf
#from keras.models import load_model
#from keras.preprocessing.image  import img_to_array
import numpy as np
from PIL import Image
import os
import uuid
from PIL import Image as PILImage

from flask import Flask, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
from ai.report_chain import generate_report 
from ai.location_fecth import get_location 

from pydantic import BaseModel
from reportlab.lib.utils import ImageReader
from huggingface_hub import hf_hub_download

from gradio_client import Client, handle_file



# ---------------------------
# App setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../git branch -M mainfrontend/templates"),
    static_folder=os.path.join(BASE_DIR, "../frontend/static")
)

app.secret_key = "skinai_secret_key"




UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    

# ---------------------------
# Classes
# ---------------------------

CLASS_NAMES =  ['Cyst', 'blackheads', 'nodules', 'papules', 'pustules', 'whiteheads']

# ---------------------------
# LOAD MODEL
# ---------------------------

model = None
'''
def load_acne_model():
    global model
    try:
        #steup no -1
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "model_with_inference_1.keras")
        
        
       #setup no 2
        

     

        model = tf.keras.models.load_model(model_path)

        print("Model loaded succesfully..")
    except Exception as e:
        print(f"Model loading error---> {e}")
        model = None

load_acne_model()
'''

# ---------------------------
# Prediction fucntion
# ---------------------------
'''
def predict_acne(img_path):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)
    pre_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])*100
    return pre_class,confidence

'''

client = Client("ParthHuggFace/modelbackedn")

def predict_acne(img_path):
    try:
        result = client.predict(
            handle_file(img_path),
            api_name="/predict"
        )

        # result comes like: {'label': 'Cyst', 'confidence': 92.3}
        acne_type = result["label"]
        confidence = result["confidence"]

        return acne_type, confidence

    except Exception as e:
        print("HF API error:", e)
        return "Error", 0

# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def home():
    return render_template("login.html")

@app.route("/login",methods = ['POST' , 'GET'])
def login():
    if request.method== "POST":

        username = request.form.get("username")
        role = request.form.get("role")
        location = request.form.get("location")

        if not role:
            return render_template("login.html" , error = "Please select the role")
    
        session['username'] = username
        session['user_type'] = role
        session['location'] = location
        
        
        return redirect(url_for("upload_img"))
    
    return render_template("login.html")


@app.route("/upload",methods = ['POST' , 'GET'])
def upload_img():
    if 'user_type' not in session:
        return redirect(url_for("home"))
    
    return render_template("upload.html")


@app.route("/process",methods = ['POST' , 'GET'])
def process_image():
    if 'user_type' not in session:
        return redirect(url_for("home"))

    image = request.files.get("image")
    if not image:
        return render_template("upload.html", error="No image uploaded")

    filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(img_path)

    acne_type, confidence = predict_acne(img_path)

    session["acne_type"] = acne_type #save acne_type
    session["confidence"] = round(confidence, 2)  #save confidence
    session["uploaded_image"] = img_path #save Image path

    if confidence < 40:
        confidence_class = "low"
    elif confidence < 70:
        confidence_class = "medium"
    else:
        confidence_class = "high"

    return render_template(
        "result.html",
        image_path=f"static/uploads/{filename}",
        acne_type=acne_type,
        confidence=confidence,
        confidence_class=confidence_class
    )

@app.route("/download-report", methods=["POST"])
def download_report():

    # ============ SAFETY ============
    if "acne_type" not in session:
        return "Please analyze image first.", 400


    # ================= LLM REPORT =================
    if "llm_report" not in session:
        llm_report = generate_report(
            topic=session.get("acne_type"),
            user_type=session.get("user_type")
        )

        if llm_report is None:
            llm_report = {
                "introduction": "",
                "causes": [],
                "symptoms": [],
                "prevention": [],
                "treatment": "",
                "conclusion": ""
            }

        if hasattr(llm_report, "dict"):
            llm_report = llm_report.dict()

        session["llm_report"] = llm_report
    else:
        llm_report = session.get("llm_report")


    # ================= HOSPITAL =================
    user_location = session.get("location", "Mumbai") 
    if "hospital_recommendation" not in session:
        hospital = get_location(
            location=user_location,
            topic="Acne"
        )

        if hasattr(hospital, "dict"):
            hospital = hospital.dict()

        session["hospital_recommendation"] = hospital
    else:
        hospital = session.get("hospital_recommendation")


    # ================= SESSION DATA =================
    report = llm_report
    acne_type = session.get("acne_type", "Unknown")
    confidence = session.get("confidence", 0)
    user_type = session.get("user_type", "Unknown")

    if not report:
        return "LLM report not found.", 400


    # ================= PDF =================
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left_margin, right_margin = 50, 50
    top_margin, bottom_margin = 50, 50
    y = height - top_margin

    from reportlab.lib.utils import simpleSplit

    def draw_multiline(text, y_pos, x_pos=left_margin, max_width=width-left_margin-right_margin):
        pdf.setFont("Helvetica", 11)
        lines = simpleSplit(str(text), "Helvetica", 11, max_width)
        for line in lines:
            pdf.drawString(x_pos, y_pos, line)
            y_pos -= 14
        return y_pos

    def draw_centered(text, y_pos, font="Helvetica-Bold", size=14):
        pdf.setFont(font, size)
        x = (width - pdf.stringWidth(text, font, size)) / 2
        pdf.drawString(x, y_pos, text)
        return y_pos - size - 6


    # ================= CONTENT =================
    y = draw_centered("SkinAI Prediction Report", y)
    image_path = session.get("uploaded_image")

    if image_path and os.path.exists(image_path):
        try:
            pil_img = PILImage.open(image_path)
            img_w, img_h = pil_img.size

            # ✅ Max allowed size in PDF
            max_width = 150
            max_height = 150

            # ✅ Scale while keeping aspect ratio
            ratio = min(max_width / img_w, max_height / img_h)
            new_width = img_w * ratio
            new_height = img_h * ratio

            img = ImageReader(image_path)

            x = (width - new_width) / 2
            y -= new_height + 10

            pdf.drawImage(
                img,
                x,
                y,
                width=new_width,
                height=new_height,
                preserveAspectRatio=True,
                mask="auto"
            )

            y -= 10

        except Exception as e:
            print("Image load error:", e)


    y = draw_multiline(f"User Type: {user_type}", y)
    y = draw_multiline(f"Detected Acne Type: {acne_type}", y)
    y = draw_multiline(f"Prediction Confidence: {confidence}%", y)
    y -= 10

    sections = ["introduction", "causes", "symptoms", "prevention", "treatment", "conclusion"]
    titles = ["Introduction", "Causes", "Symptoms & Appearance", "Prevention", "Treatment Plan", "Conclusion"]

    for title, key in zip(titles, sections):
        y = draw_centered(title, y, size=12)
        content = report.get(key, "")
        if isinstance(content, list):
            for item in content:
                y = draw_multiline(f"- {item}", y)
        else:
            y = draw_multiline(content, y)
        y -= 6


    # ================= HOSPITAL =================
    y = draw_centered("Recommended Dermatology Hospital", y, size=12)

    if hospital:
        y = draw_multiline(f"Name: {hospital.get('Name', 'N/A')}", y)
        y = draw_multiline(f"Area: {hospital.get('Area', 'N/A')}", y)
        y = draw_multiline(f"Reason: {hospital.get('Reason', '')}", y)
    else:
        y = draw_multiline("No hospital recommendation available.", y)


    # ================= DISCLAIMER =================
    y = draw_centered("Disclaimer", y, size=12)
    y = draw_multiline(
        "This report is AI-generated and intended for informational purposes only. "
        "It does not replace professional medical advice.",
        y
    )

    pdf.showPage()
    pdf.save()
    buffer.seek(0)


    # ================= CLEAR SESSION =================
    session.pop("llm_report", None)
    session.pop("hospital_recommendation", None)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="SkinAI_Detailed_Report.pdf",
        mimetype="application/pdf"
    )




@app.route("/result",methods=['GET'])
def result():
    return render_template("result.html")
        

@app.route("/contact",methods=['GET'])
def contact():
    return render_template("contact.html")
        



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
