import os
import json
from fastapi import FastAPI, Form, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag_module import MedicalRAGEngine
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app ONCE at the top with your title
app = FastAPI(title="AI Copilot Communications Gateway")

# Configure CORS right after initializing the app
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
    
    # Flatten all categories into a single comprehensive list of 300+ items
    complete_inventory = []
    for category_list in data.values():
        complete_inventory.extend(category_list)
        
    return {"medicines": complete_inventory, "total_count": len(complete_inventory)}

@app.post("/api/checkout")
async def simulate_stripe_payment(doctor_id: str, amount: float):
    # Simulated Stripe Checkout & Appointment Token Generation
    return {
        "status": "success",
        "transaction_id": f"txn_medisync_{os.urandom(4).hex()}",
        "amount_paid": amount,
        "token": f"MS-PASS-{os.urandom(3).hex().upper()}",
        "message": "Payment verified successfully via Stripe simulator."
    }

@app.post("/sms/incoming")
async def handle_incoming_query(Body: str = Form(...), From: str = Form(...)):
    """
    Receives text queries regarding reports, queries the internal Vector DB,
    and returns a contextual message.
    """
    print(f"Received message from {From}: {Body}")

    # 1. Query Vector store for relevant medical logic
    context = rag_engine._retrieve_medical_context(Body)

    # 2. Run an agent conversation layer using Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

    agent_prompt = (
        f"You are a helpful medical assistant agent answering a user query over a messaging interface.\n"
        f"User Query: {Body}\n\n"
        f"Relevant Guidelines:\n{context}\n\n"
        f"Provide a concise, clear answer suitable for text messaging. Keep it brief and professional."
    )

    response = llm.invoke(agent_prompt)

    # Return response payload structured for webhooks
    return {
        "recipient": From,
        "reply_message": response.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)