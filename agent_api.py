import os
import json
import io
from fastapi import FastAPI, Form, UploadFile, File, Response
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag_module import MedicalRAGEngine
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 1. Initialize FastAPI app ONCE at the top
app = FastAPI(title="AI Copilot Communications Gateway")

# 2. Configure CORS right after initializing the app
origins = [
    "https://medisync-design-production.up.railway.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize RAG engine so it's ready for endpoints
rag_engine = MedicalRAGEngine(api_key=api_key)

# Define the 12 Specialized Doctor Personas
DOCTOR_PERSONAS = [
    {"id": "cardio", "name": "Dr. Aris Thorne", "specialty": "Cardiology", "fee": 50, "bio": "Expert in heart arrhythmias and ECG anomalies."},
    {"id": "neuro", "name": "Dr. Elena Rostova", "specialty": "Neurology", "fee": 60, "bio": "Specialized in migraines and neuro-tracking."},
    {"id": "derm", "name": "Dr. Marcus Vance", "specialty": "Dermatology", "fee": 40, "bio": "Focuses on skin lesion classifications and biopsies."},
    {"id": "ortho", "name": "Dr. Sarah Jenkins", "specialty": "Orthopedics", "fee": 45, "bio": "Expert in joint fractures and ligament analysis."},
    {"id": "pedi", "name": "Dr. Lucas Chen", "specialty": "Pediatrics", "fee": 40, "bio": "Dedicated child healthcare and developmental milestones."},
    {"id": "endo", "name": "Dr. Nadia Al-Mansoor", "specialty": "Endocrinology", "fee": 55, "bio": "Specializes in metabolic disorders and thyroid profiles."},
    {"id": "ophthal", "name": "Dr. Julian Sterling", "specialty": "Ophthalmology", "fee": 45, "bio": "Expert in retinal scans and intraocular pressure."},
    {"id": "psych", "name": "Dr. Maya Lin", "specialty": "Psychiatry", "fee": 65, "bio": "Provides psychological evaluation and stress management."},
    {"id": "gastro", "name": "Dr. Omar Farooq", "specialty": "Gastroenterology", "fee": 50, "bio": "Specializes in digestive tracts and liver panels."},
    {"id": "pulmo", "name": "Dr. Clara Oswald", "specialty": "Pulmonology", "fee": 50, "bio": "Expert in chest X-rays and pulmonary function testing."},
    {"id": "nephro", "name": "Dr. David Kim", "specialty": "Nephrology", "fee": 55, "bio": "Specializes in kidney stone detection and urinalysis."},
    {"id": "onco", "name": "Dr. Beatrice Vane", "specialty": "Oncology", "fee": 80, "bio": "Advanced diagnostic specialist reviewing complex multi-stage scans."}
]

@app.get("/api/doctors")
async def get_doctors():
    return DOCTOR_PERSONAS

@app.get("/api/medicines")
async def get_all_medicines(specialty: Optional[str] = None):
    """
    Serves medicines dynamically from medicines_db.json.
    Supports filtering by specialty category name.
    """
    db_path = "medicines_db.json"
    if not os.path.exists(db_path):
        return {"medicines": [], "total_count": 0}
        
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if specialty and specialty in data:
        return {"medicines": data[specialty], "total_count": len(data[specialty])}
    
    complete_inventory = []
    for category_list in data.values():
        complete_inventory.extend(category_list)
        
    return {"medicines": complete_inventory, "total_count": len(complete_inventory)}

@app.post("/api/checkout")
async def simulate_stripe_payment(doctor_id: str, amount: float):
    return {
        "status": "success",
        "transaction_id": f"txn_medisync_{os.urandom(4).hex()}",
        "amount_paid": amount,
        "token": f"MS-PASS-{os.urandom(3).hex().upper()}",
        "message": "Payment verified successfully via Stripe simulator."
    }

@app.post("/api/prescription/pdf")
async def generate_prescription_pdf(data: dict):
    """
    Generates a professional PDF prescription matching the template layout,
    complete with a faint transparent watermark logo in the background,
    correct doctor profile, patient metadata, and structured clinical notes.
    """
    try:
        doctor_id = data.get("doctorId", "cardio")
        diagnosis = data.get("diagnosis", "Clinical evaluation completed.")

        # Find the matching doctor persona dictionary
        selected_doc = next((doc for doc in DOCTOR_PERSONAS if doc["id"] == doctor_id), DOCTOR_PERSONAS[0])

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter # 612 x 792 points

        # 1. Left Side Blue Branding Panel (Hospital Name Sidebar)
        p.setFillColorRGB(0.22, 0.51, 0.96) # Professional medical blue (#3b82f6)
        p.rect(0, 0, 130, height, fill=1, stroke=0)

        # Vertical Hospital Name Text
        p.saveState()
        p.setFillColorRGB(1, 1, 1)
        p.setFont("Helvetica-Bold", 22)
        p.translate(55, height / 2 - 40)
        p.rotate(90)
        p.drawString(0, 0, "MEDISYNC CLINICAL")
        p.restoreState()

        # 2. Transparent Background Watermark Logo
        p.saveState()
        p.setFont("Helvetica-Bold", 55)
        # Using a very light grey/blue with low opacity effect for the watermark
        p.setFillColorRGB(0.88, 0.92, 0.98) 
        p.translate(width / 2 + 30, height / 2 - 50)
        p.rotate(30)
        p.drawCentredString(0, 0, "MEDISYNC")
        p.restoreState()

        # 3. Header Area (Doctor Information)
        p.setFillColorRGB(0.1, 0.1, 0.1)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(155, height - 50, f"{selected_doc['name']}")
        
        p.setFont("Helvetica", 10)
        p.setFillColorRGB(0.4, 0.4, 0.4)
        p.drawString(155, height - 68, f"{selected_doc['specialty']} Specialist — Fee: ${selected_doc['fee']}")
        
        # Serial Number & Date
        p.setFont("Helvetica-Bold", 9)
        p.drawString(width - 110, height - 50, "Sr.# 2026-094")
        p.setFont("Helvetica", 9)
        p.drawString(width - 110, height - 65, "Date: 2026-07-26")

        # Top Header Divider Line
        p.setStrokeColorRGB(0.8, 0.8, 0.8)
        p.setLineWidth(1)
        p.line(155, height - 85, width - 40, height - 85)

        # 4. Patient Meta Fields (Name, Age, Gender, Weight)
        p.setFont("Helvetica-Bold", 9)
        p.setFillColorRGB(0.2, 0.2, 0.2)
        p.drawString(155, height - 110, "Patient Name:")
        p.line(225, height - 113, width - 40, height - 113)

        p.drawString(155, height - 132, "Age: ______")
        p.drawString(255, height - 132, "Gender: ________")
        p.drawString(380, height - 132, "Weight: ________")

        # 5. Rx Symbol Area
        p.setFont("Helvetica-Bold", 28)
        p.setFillColorRGB(0.22, 0.51, 0.96)
        p.drawString(155, height - 180, "R")
        p.setFont("Helvetica-Bold", 18)
        p.drawString(174, height - 177, "x")

        # 6. Clinical Findings / Diagnosis & Medication Summary Section
        p.setFont("Helvetica-Bold", 11)
        p.setFillColorRGB(0.1, 0.1, 0.1)
        p.drawString(155, height - 215, "Diagnosis & Clinical Summary:")

        p.setFont("Helvetica", 10)
        p.setFillColorRGB(0.2, 0.2, 0.2)
        text_object = p.beginText(155, height - 235)
        text_object.setLeading(15)

        for line in diagnosis.split('\n'):
            clean_line = line.replace('**', '')
            if len(clean_line) > 72:
                chunks = [clean_line[i:i+72] for i in range(0, len(clean_line), 72)]
                for chunk in chunks:
                    text_object.textLine(chunk)
            else:
                text_object.textLine(clean_line)
        
        p.drawText(text_object)

        # 7. Footer Signature & Clinic Contact Details
        p.setFont("Helvetica", 9)
        p.setFillColorRGB(0.4, 0.4, 0.4)
        p.drawString(width - 180, 95, "__________________________")
        p.setFont("Helvetica-Bold", 9)
        p.drawString(width - 155, 80, "Doctor's Signature")

        p.setStrokeColorRGB(0.8, 0.8, 0.8)
        p.line(155, 60, width - 40, 60)

        p.setFont("Helvetica", 8)
        p.drawString(155, 45, "📍 Medisync Telehealth Tower, Dhaka")
        p.drawString(330, 45, "📞 +880-9612-MEDISYNC")
        p.drawString(480, 45, "✉ support@medisync.app")

        p.showPage()
        p.save()

        buffer.seek(0)
        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Medisync_Prescription_{doctor_id}.pdf"}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/chat")
async def handle_chat_message(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Handles live telehealth chat messages with a warm, human-like medical specialist persona,
    evaluates knowledge base context, and suggests seeing a real physician if details are missing.
    """
    try:
        response_text = ""
        
        # If an image/scan is attached, process it with the vision module or RAG
        if file:
            image_bytes = await file.read()
            from vision_module import MedicalVisionEngine
            vision_engine = MedicalVisionEngine()
            vision_res = vision_engine.analyze_scan(image_bytes, api_key=api_key)
            
            if vision_res.get("success") and vision_res.get("findings"):
                finding = vision_res["findings"][0]
                report = rag_engine.generate_doctor_report(
                    finding=finding["class_name"],
                    size_mm=finding.get("estimated_size_mm", 5.0),
                    location=finding.get("location_tags", "General"),
                    api_key=api_key
                )
                response_text = f"I've taken a close look at your attached scan. Here is what stands out from our clinical evaluation:\n\n{report}"
            else:
                response_text = "I received your attached scan, but I don't see any definitive anomalies standing out. Let's talk about how you're feeling physically right now."
        else:
            # Standard text-based RAG query with human conversational persona
            context = rag_engine._retrieve_medical_context(message)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.4)
            
            prompt = (
                f"You are a warm, empathetic, and professional human medical specialist doctor talking directly with your patient in a live telehealth session.\n"
                f"Avoid sounding like a cold robotic rule-engine or listing massive safety disclaimers unless it is a life-threatening emergency.\n"
                f"Patient Message: {message}\n\n"
                f"Retrieved Medical Guidelines / Knowledge Base Context:\n{context}\n\n"
                f"Instructions:\n"
                f"1. Speak naturally like a caring human doctor.\n"
                f"2. Check if the retrieved knowledge base context contains enough solid information to answer the user's specific query.\n"
                f"3. If the vector database / guidelines context does NOT have sufficient information to safely or accurately answer the patient's specific question, or if it's a complex/critical medical scenario, explicitly tell the user: 'I don't have enough specific details in our medical reference database for this, so you should schedule a consultation with a real human physician for a thorough evaluation.'\n"
                f"4. Keep your response conversational, concise, and supportive."
            )
            res = llm.invoke(prompt)
            response_text = res.content

        return {
            "status": "success",
            "reply": response_text
        }
    except Exception as e:
        return {
            "status": "error",
            "reply": f"I apologize, but I ran into a minor snag on our server end: {str(e)}"
        }

@app.post("/sms/incoming")
async def handle_incoming_query(Body: str = Form(...), From: str = Form(...)):
    print(f"Received message from {From}: {Body}")
    context = rag_engine._retrieve_medical_context(Body)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

    agent_prompt = (
        f"You are a helpful medical assistant agent answering a user query over a messaging interface.\n"
        f"User Query: {Body}\n\n"
        f"Relevant Guidelines:\n{context}\n\n"
        f"Provide a concise, clear answer suitable for text messaging. Keep it brief and professional."
    )

    response = llm.invoke(agent_prompt)

    return {
        "recipient": From,
        "reply_message": response.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)