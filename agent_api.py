import os
import json
from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag_module import MedicalRAGEngine
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

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

@app.post("/api/chat")
async def handle_chat_message(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Handles live telehealth chat messages, analyzes optional image uploads using Vision/RAG,
    and returns a clinical AI response.
    """
    try:
        response_text = ""
        
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
                response_text = f"I have analyzed your uploaded scan.\n\n{report}"
            else:
                response_text = "I received your image scan, but no clear clinical anomalies met the confidence threshold. Let's discuss your symptoms."
        else:
            context = rag_engine._retrieve_medical_context(message)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
            prompt = (
                f"You are an assigned Medisync specialist doctor persona.\n"
                f"User Query: {message}\n\n"
                f"Medical Guidelines Context:\n{context}\n\n"
                f"Provide a professional, clinical, and helpful response."
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
            "reply": f"Clinical server encountered an error: {str(e)}"
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