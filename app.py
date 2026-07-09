import os
import sys
import datetime
import sqlite3
import threading
import logging
import traceback
from logging.handlers import RotatingFileHandler
from functools import wraps
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.helper import download_hugging_face_embeddings, text_split, load_file_by_type
# from src.prompt import system_prompt # Using defined prompt below for RAG context
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import requests

# ─── LOGGER SETUP ────────────────────────────────────────────────────────────
_log_fmt = logging.Formatter(
    fmt="%(asctime)s  [%(levelname)-8s]  %(funcName)s():%(lineno)d  —  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curabot.log")
_file_handler = RotatingFileHandler(_log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_file_handler.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_fmt)
_console_handler.setLevel(logging.INFO)

logger = logging.getLogger("curabot")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
logger.propagate = False          # Don't double-print via root logger
# ─────────────────────────────────────────────────────────────────────────────

# 1. LOAD CONFIG & PATHS
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)
# IMPORTANT: Change this fallback key in production or set it in .env
app.secret_key = os.getenv("FLASK_SECRET_KEY", "curabot-super-fallback-key-999")

# 2. FILE UPLOAD CONFIGURATION
UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PRESCRIPTION_FOLDER = os.path.join(base_dir, 'static', 'prescriptions')
os.makedirs(PRESCRIPTION_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max (large medical PDFs)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
KB_ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv', 'png', 'jpg', 'jpeg'}
PRESCRIPTION_ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 3. SQLITE DATABASE SETUP WITH PERSISTENT USER PROFILES
DB_PATH = os.path.join(base_dir, 'curabot.db')

def init_db():
    """Initializes SQLite database with required tables, including users."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Admin User Accounts Table (persistent logins)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Documents registry table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT DEFAULT 'Audited'
        )
    ''')

    # Symptom trends table
    c.execute('''
        CREATE TABLE IF NOT EXISTS symptom_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptom TEXT NOT NULL,
            logged_at TEXT NOT NULL
        )
    ''')

    # Chat history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    # Knowledge base ingestion registry
    c.execute('''
        CREATE TABLE IF NOT EXISTS kb_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'processing',
            error_message TEXT,
            indexed_at TEXT,
            file_size INTEGER DEFAULT 0
        )
    ''')

    # Hospitals / Doctors registry (fed by admin)
    c.execute('''
        CREATE TABLE IF NOT EXISTS hospitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_name TEXT NOT NULL,
            doctor_name TEXT,
            specialization TEXT NOT NULL,
            location TEXT,
            phone TEXT,
            available_days TEXT DEFAULT 'Mon,Tue,Wed,Thu,Fri',
            time_slots TEXT DEFAULT '09:00 AM,10:00 AM,11:00 AM,02:00 PM,03:00 PM,04:00 PM',
            is_active INTEGER DEFAULT 1,
            created_at TEXT
        )
    ''')

    # Appointment booking table
    c.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_age INTEGER,
            patient_phone TEXT,
            specialization TEXT NOT NULL,
            hospital_id INTEGER,
            hospital_name TEXT,
            doctor_name TEXT,
            preferred_date TEXT NOT NULL,
            preferred_time TEXT,
            symptoms TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TEXT
        )
    ''')

    # Doctors table — multiple doctors per hospital
    c.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            hospital_name TEXT NOT NULL,
            doctor_name TEXT NOT NULL,
            specialization TEXT NOT NULL,
            qualification TEXT DEFAULT '',
            experience_years INTEGER DEFAULT 0,
            time_slots TEXT DEFAULT '09:00 AM,10:00 AM,11:00 AM,02:00 PM,03:00 PM,04:00 PM',
            available_days TEXT DEFAULT 'Mon,Tue,Wed,Thu,Fri',
            is_active INTEGER DEFAULT 1,
            created_at TEXT
        )
    ''')

    # Migrate existing appointments table — add new columns if missing
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(appointments)").fetchall()}
    for col, defn in [
        ("hospital_id", "INTEGER"), ("hospital_name", "TEXT"), ("doctor_name", "TEXT"),
        ("doctor_id", "INTEGER"),
        ("patient_gender", "TEXT"), ("patient_blood_group", "TEXT"),
        ("patient_weight", "TEXT"), ("patient_height", "TEXT"),
        ("patient_allergies", "TEXT"), ("patient_address", "TEXT"),
        ("patient_email", "TEXT"), ("doctor_note", "TEXT")
    ]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE appointments ADD COLUMN {col} {defn}")

    # Patient accounts table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    ''')

    # Blood bank request table
    c.execute('''
        CREATE TABLE IF NOT EXISTS blood_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_age INTEGER,
            patient_phone TEXT,
            hospital_name TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            units_needed INTEGER DEFAULT 1,
            urgency TEXT DEFAULT 'Normal',
            prescription_filename TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TEXT
        )
    ''')

    # Blood donors table
    c.execute('''
        CREATE TABLE IF NOT EXISTS blood_donors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            city TEXT NOT NULL,
            is_available INTEGER DEFAULT 1,
            created_at TEXT
        )
    ''')

    # SEED TEST USER (If users space is completely empty)
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        # Sets default account: admin@curabot.com / password123
        hashed_pw = generate_password_hash("password123")
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", ("admin@curabot.com", hashed_pw))
        logger.info("Seeded default admin account: admin@curabot.com / password123")

    conn.commit()
    conn.close()
    logger.info("SQLite DB initialized — all tables ready.")

init_db()

# DB HELPER CODES (Required for app logic)
def db_log_document(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename, timestamp, status) VALUES (?, ?, ?)",
              (filename, datetime.datetime.now().strftime("%I:%M %p"), "Audited"))
    conn.commit()
    conn.close()

def db_get_documents():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename, timestamp, status FROM documents ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"filename": r[0], "timestamp": r[1], "status": r[2]} for r in rows]

def db_log_symptom(symptom):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO symptom_log (symptom, logged_at) VALUES (?, ?)", (symptom, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def db_get_symptom_trends():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT symptom, COUNT(*) as count FROM symptom_log GROUP BY symptom ORDER BY count DESC")
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}

def db_save_chat_message(session_id, role, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, message, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def db_get_chat_history(session_id, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, message FROM chat_history WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

# 4. EMERGENCY KEYWORD CONFIGS
EMERGENCY_KEYWORDS = ["chest pain", "heart attack", "can't breathe", "difficulty breathing", "stroke", "seizure"]
def check_emergency(msg):
    msg_lower = msg.lower()
    return any(kw in msg_lower for kw in EMERGENCY_KEYWORDS)

EMERGENCY_RESPONSE = "⚠️ URGENT: Symptoms match medical emergency profile guidelines. Please dial localized emergency hotlines (112 / 911) right away."

TRACKED_SYMPTOMS = ["headache", "fever", "nausea", "fatigue", "cough", "anxiety"]
def extract_and_log_symptoms(msg):
    msg_lower = msg.lower()
    for symptom in TRACKED_SYMPTOMS:
        if symptom in msg_lower:
            db_log_symptom(symptom)

def extract_pdf_text(file_path):
    try:
        import pdfplumber
        text_content = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: text_content.append(page_text.strip())
        return "\n".join(text_content) if text_content else None
    except Exception: return None

# 5. DATABASE SESSIONS DECORATOR
def require_admin_session(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return jsonify({"error": "Unauthorized session context layout."}), 401
        return f(*args, **kwargs)
    return decorated

if not os.getenv("GROQ_API_KEY"):
    logger.critical("GROQ_API_KEY is missing — server cannot start.")
    raise ValueError("GROQ_API_KEY Missing!")

SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
logger.info("Loading embedding model (BAAI/bge-base-en-v1.5)...")
embeddings = download_hugging_face_embeddings()
logger.info("Connecting to local Qdrant vector database...")

_COLLECTION_NAME   = "medical_chatbot_base"  # bge-base-en-v1.5 collection
_BGE_M3_DIM        = 768   # bge-base-en-v1.5 output dimension

client = QdrantClient(host="localhost", port=6333)

# Create collection if it doesn't exist yet
try:
    _col_info     = client.get_collection(_COLLECTION_NAME)
    _existing_dim = _col_info.config.params.vectors.size
    if _existing_dim != _BGE_M3_DIM:
        logger.warning("Dimension mismatch (%d vs %d) — recreating collection.", _existing_dim, _BGE_M3_DIM)
        client.delete_collection(_COLLECTION_NAME)
        raise Exception("recreate")
    logger.info("Collection '%s' ready (dim=%d).", _COLLECTION_NAME, _existing_dim)
except Exception:
    client.create_collection(
        collection_name=_COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=_BGE_M3_DIM,
            distance=qdrant_models.Distance.COSINE
        )
    )
    logger.info("Collection '%s' created fresh with dim=%d.", _COLLECTION_NAME, _BGE_M3_DIM)

vector_store    = QdrantVectorStore(client=client, collection_name=_COLLECTION_NAME, embedding=embeddings)
_dense_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
logger.info("Qdrant vector store ready (bge-base-en-v1.5, dim=%d).", _BGE_M3_DIM)

# Build BM25 sparse index from all stored chunks
logger.info("Building BM25 sparse index from Qdrant collection (this may take ~60s)...")
_all_docs = []
_scroll_offset = None
while True:
    _results, _scroll_offset = client.scroll(
        collection_name=_COLLECTION_NAME,
        limit=500,
        offset=_scroll_offset,
        with_payload=True,
        with_vectors=False,
    )
    for point in _results:
        _text = point.payload.get("page_content") or point.payload.get("document", "")
        _meta = point.payload.get("metadata", {})
        if _text:
            _all_docs.append(Document(page_content=_text, metadata=_meta))
    if _scroll_offset is None:
        break
logger.info("BM25 index built from %d chunks.", len(_all_docs))
_bm25_retriever  = BM25Retriever.from_documents(_all_docs, k=15)

# Hybrid: 60% dense + 40% BM25 with Reciprocal Rank Fusion
_hybrid_retriever = EnsembleRetriever(
    retrievers=[_dense_retriever, _bm25_retriever],
    weights=[0.6, 0.4],
)

# Reranker on top of hybrid results
logger.info("Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2)...")
_rerank_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
_compressor   = CrossEncoderReranker(model=_rerank_model, top_n=5)
retriever     = ContextualCompressionRetriever(base_compressor=_compressor, base_retriever=_hybrid_retriever)
logger.info("Hybrid retriever ready — BM25(40%%) + Dense(60%%) → reranked to top 5.")

logger.info("Initializing ChatGroq LLM (llama-3.1-8b-instant)...")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=1000)


# LangChain RAG prompt structure
system_prompt = (
    "You are CuraBot, an empathetic personal medical AI assistant styled as a friendly AI Medical Specialist.\n"
    "Your tone must be warm, supportive, grounded, and deeply conversational — like a caring healthcare peer.\n\n"
    "STRICT SCOPE RULE — READ FIRST:\n"
    "You ONLY answer questions related to medicine, health, symptoms, diseases, treatments, medications, anatomy, or wellness.\n"
    "If the user's question is NOT medical in nature (e.g. politics, sports, general knowledge, current events, entertainment, coding, etc.), "
    "you must REFUSE to answer it. Respond with exactly: "
    "'I am CuraBot, a medical AI assistant. I can only help with health and medical questions. "
    "Please consult a general-purpose assistant for non-medical topics.'\n"
    "Do NOT attempt to answer non-medical questions under any circumstances, even if the retrieved context is empty.\n\n"
    "CORE INTERACTION RULES:\n"
    "1. Validate feelings first: When a user mentions a symptom, start by empathetically acknowledging how they feel.\n"
    "2. Strict answer scope: ONLY answer what the user is explicitly asking. Do not dump unrequested lists.\n"
    "3. Prescription context: If a prescription or document is referenced, acknowledge it warmly and interpret the user's question around it.\n"
    "4. Cite sources: When answering from medical knowledge, mention the relevant source book where appropriate.\n"
    "5. Formatting: Use point-wise, scannable responses. No raw markdown symbols like ## or **.\n"
    "6. Safety: Never diagnose definitively. Always recommend consulting a doctor for serious concerns.\n"
    "7. Confidence note: If the retrieved context doesn't clearly address the question, honestly say so.\n\n"
    "Retrieved Medical Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = os.urandom(16).hex()
    return render_template('chat.html')

# 🔐 NEW AUTH ENDPOINT: Validates against SQLite users table
@app.route("/admin/login", methods=["POST"])
def admin_login_api():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if not email or not password:
        return jsonify({"success": False, "message": "Missing credentials."}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()

    if row and check_password_hash(row[0], password):
        session["admin_logged_in"] = True
        session["admin_email"] = email
        logger.info("Admin login SUCCESS: %s", email)
        return jsonify({"success": True, "email": email})

    logger.warning("Admin login FAILED for email: %s", email)
    return jsonify({"success": False, "message": "Invalid email or password."}), 401

@app.route("/admin/logout", methods=["POST"])
def admin_logout_api():
    session.pop("admin_logged_in", None)
    session.pop("admin_email", None)
    return jsonify({"success": True})

# 👤 NEW REGISTRATION ENDPOINT: Harshes and stores a new admin account
@app.route("/admin/register", methods=["POST"])
@require_admin_session # Optional: Require existing admin login to create new ones
def admin_register_api():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    
    if not email or not password:
        return jsonify({"success": False, "message": "Missing email or password."}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed_pw = generate_password_hash(password)
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "User registered successfully!"})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email already exists."}), 400

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    session_id = session.get("session_id", "default")

    if not msg and 'file' not in request.files:
        return jsonify({"error": "Missing input."}), 400

    if msg and check_emergency(msg):
        db_save_chat_message(session_id, "user", msg)
        db_save_chat_message(session_id, "assistant", EMERGENCY_RESPONSE)
        return EMERGENCY_RESPONSE

    file_context_text = ""
    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            db_log_document(filename)
            file_context_text = f"\n\n[Uploaded File Attachment: {filename}]"

    if msg: extract_and_log_symptoms(msg)

    combined_input = f"{msg}{file_context_text}".strip()
    history_rows = db_get_chat_history(session_id, limit=10)
    chat_history = [HumanMessage(content=m) if r == "user" else AIMessage(content=m) for r, m in history_rows]

    try:
        response = rag_chain.invoke({"input": combined_input, "chat_history": chat_history})
        answer = response["answer"]
    except Exception:
        answer = "Error handling calculation context parameters."

    db_save_chat_message(session_id, "user", combined_input)
    db_save_chat_message(session_id, "assistant", answer)
    return answer

@app.route("/admin/metadata", methods=["GET"])
@require_admin_session # Standard session security validation
def get_admin_metadata():
    """Returns detailed audit trails for the protected workspace view."""
    documents = db_get_documents()
    symptom_trends = db_get_symptom_trends()

    # Most reported symptom aggregation for dashboard highlight card
    top_symptom = max(symptom_trends, key=symptom_trends.get) if symptom_trends else "N/A"
    top_symptom_count = symptom_trends.get(top_symptom, 0)

    return jsonify({
        "prescription_count": len(documents),
        "registry": documents, # Detailed table rows
        "symptom_trends": symptom_trends,
        "top_symptom": {
            "name": top_symptom.title(),
            "count": top_symptom_count
        }
    })

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear conversation history for the current user session."""
    session_id = session.get("session_id", "default")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Chat history cleared successfully."})

@app.errorhandler(413)
def file_too_large(_e):
    return jsonify({"success": False, "message": "File too large. Maximum allowed size is 500 MB."}), 413


def _process_document_background(doc_id, file_path, original_filename):
    """Runs in a daemon thread — load, chunk, embed, upsert into Qdrant."""
    logger.info("[doc_id=%d] START processing: %s", doc_id, original_filename)
    try:
        logger.debug("[doc_id=%d] Step 1/4 — Loading file from: %s", doc_id, file_path)
        documents = load_file_by_type(file_path)
        if not documents:
            raise Exception("No content could be extracted from the file.")
        logger.debug("[doc_id=%d] Step 1/4 DONE — loaded %d document object(s)", doc_id, len(documents))

        logger.debug("[doc_id=%d] Step 2/4 — Splitting into chunks...", doc_id)
        chunks = text_split(documents)
        chunk_count = len(chunks)
        if chunk_count == 0:
            raise Exception("File contains no indexable text after splitting.")
        logger.info("[doc_id=%d] Step 2/4 DONE — %d chunks created", doc_id, chunk_count)

        logger.info("[doc_id=%d] Step 3/4 — Embedding & upserting to Qdrant in batches...", doc_id)
        _BATCH_SIZE = 50
        for batch_start in range(0, chunk_count, _BATCH_SIZE):
            batch = chunks[batch_start: batch_start + _BATCH_SIZE]
            vector_store.add_documents(batch)
            logger.debug("[doc_id=%d] Upserted %d/%d chunks",
                         doc_id, min(batch_start + _BATCH_SIZE, chunk_count), chunk_count)
        logger.info("[doc_id=%d] Step 3/4 DONE — all %d chunks stored in Qdrant", doc_id, chunk_count)

        logger.debug("[doc_id=%d] Step 4/4 — Updating DB status to indexed", doc_id)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE kb_documents SET status='indexed', chunk_count=?, indexed_at=? WHERE id=?",
                  (chunk_count, datetime.datetime.now().isoformat(), doc_id))
        conn.commit()
        conn.close()
        logger.info("[doc_id=%d] COMPLETE — '%s' indexed with %d chunks", doc_id, original_filename, chunk_count)

    except Exception as e:
        logger.error("[doc_id=%d] FAILED — %s: %s", doc_id, original_filename, e)
        logger.debug("[doc_id=%d] Traceback:\n%s", doc_id, traceback.format_exc())
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE kb_documents SET status='failed', error_message=? WHERE id=?",
                  (str(e)[:500], doc_id))
        conn.commit()
        conn.close()


@app.route("/admin/ingest", methods=["POST"])
@require_admin_session
def admin_ingest():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file provided."}), 400

        file = request.files['file']
        if not file or not file.filename:
            return jsonify({"success": False, "message": "Empty file."}), 400

        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in KB_ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "message": f"File type .{ext} not supported. Allowed: pdf, docx, xlsx, csv, png, jpg, jpeg"}), 400

        original_filename = file.filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"success": False, "message": "Invalid filename."}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        logger.info("File saved: %s (%.2f MB)", filename, file_size / 1024 / 1024)

        # Reset stale record for same filename if exists (re-upload scenario)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM kb_documents WHERE filename=? AND status IN ('processing', 'failed')", (filename,))
        c.execute(
            "INSERT INTO kb_documents (filename, original_filename, file_type, chunk_count, status, indexed_at, file_size) VALUES (?, ?, ?, 0, 'processing', ?, ?)",
            (filename, original_filename, ext, datetime.datetime.now().isoformat(), file_size)
        )
        doc_id = c.lastrowid
        conn.commit()
        conn.close()

        # Fire background thread — return immediately to browser
        logger.info("Spawning background thread for doc_id=%d (%s)", doc_id, original_filename)
        t = threading.Thread(
            target=_process_document_background,
            args=(doc_id, file_path, original_filename),
            daemon=True
        )
        t.start()

        return jsonify({
            "success": True,
            "processing": True,
            "doc_id": doc_id,
            "message": f"'{original_filename}' is being processed in the background. The table will auto-refresh every 5 seconds."
        })

    except Exception as outer_err:
        logger.error("Ingest route unhandled exception: %s\n%s", outer_err, traceback.format_exc())
        return jsonify({"success": False, "message": f"Unexpected error: {str(outer_err)}"}), 500


@app.route("/admin/kb/stats", methods=["GET"])
@require_admin_session
def admin_kb_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) FROM kb_documents WHERE status='indexed'")
    doc_count, total_chunks = c.fetchone()
    c.execute("SELECT indexed_at FROM kb_documents WHERE status='indexed' ORDER BY indexed_at DESC LIMIT 1")
    last_row = c.fetchone()
    last_indexed = last_row[0] if last_row else None
    conn.close()

    try:
        collection_info = client.get_collection("medical_chatbot")
        qdrant_count = collection_info.points_count
    except Exception:
        qdrant_count = 0

    return jsonify({
        "indexed_docs": doc_count,
        "total_chunks": int(total_chunks),
        "last_indexed": last_indexed,
        "qdrant_total_vectors": qdrant_count
    })


@app.route("/admin/kb/documents", methods=["GET"])
@require_admin_session
def admin_kb_documents():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, original_filename, file_type, chunk_count, status, error_message, indexed_at, file_size FROM kb_documents ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    docs = [{
        "id": r[0], "filename": r[1], "file_type": r[2], "chunk_count": r[3],
        "status": r[4], "error": r[5], "indexed_at": r[6], "file_size": r[7]
    } for r in rows]

    return jsonify({"documents": docs})


@app.route("/admin/kb/documents/<int:doc_id>", methods=["DELETE"])
@require_admin_session
def admin_kb_delete_document(doc_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename, status FROM kb_documents WHERE id=?", (doc_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "message": "Document not found."}), 404

    filename, status = row
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if status == 'indexed':
        try:
            client.delete(
                collection_name=_COLLECTION_NAME,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[qdrant_models.FieldCondition(
                            key="metadata.source",
                            match=qdrant_models.MatchValue(value=file_path)
                        )]
                    )
                )
            )
        except Exception as e:
            print(f"Qdrant deletion warning: {e}")

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"File deletion warning: {e}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM kb_documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Document removed from knowledge base."})


# ─── APPOINTMENT ROUTES ───────────────────────────────────────────────────────

@app.route("/hospitals", methods=["GET"])
def get_hospitals():
    spec = request.args.get("specialization", "").strip()
    date = request.args.get("date", "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if spec:
        c.execute("SELECT id,hospital_name,doctor_name,specialization,location,phone,available_days,time_slots FROM hospitals WHERE is_active=1 AND specialization=? ORDER BY hospital_name", (spec,))
    else:
        c.execute("SELECT id,hospital_name,doctor_name,specialization,location,phone,available_days,time_slots FROM hospitals WHERE is_active=1 ORDER BY specialization,hospital_name")
    rows = c.fetchall()
    keys = ["id","hospital_name","doctor_name","specialization","location","phone","available_days","time_slots"]
    hospitals = [dict(zip(keys, r)) for r in rows]
    if date:
        for h in hospitals:
            c.execute(
                "SELECT preferred_time FROM appointments WHERE hospital_id=? AND preferred_date=? AND status NOT IN ('Cancelled')",
                (h["id"], date)
            )
            h["booked_slots"] = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify(hospitals)


@app.route("/admin/hospitals", methods=["GET"])
@require_admin_session
def admin_list_hospitals():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,hospital_name,doctor_name,specialization,location,phone,available_days,time_slots,is_active,created_at FROM hospitals ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    keys = ["id","hospital_name","doctor_name","specialization","location","phone","available_days","time_slots","is_active","created_at"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@app.route("/admin/hospitals", methods=["POST"])
@require_admin_session
def admin_add_hospital():
    data = request.get_json()
    name  = (data.get("hospital_name") or "").strip()
    doc   = (data.get("doctor_name") or "").strip()
    spec  = (data.get("specialization") or "").strip()
    loc   = (data.get("location") or "").strip()
    phone = (data.get("phone") or "").strip()
    days  = (data.get("available_days") or "Mon,Tue,Wed,Thu,Fri").strip()
    slots = (data.get("time_slots") or "09:00 AM,10:00 AM,11:00 AM,02:00 PM,03:00 PM,04:00 PM").strip()
    if not name or not spec:
        return jsonify({"success": False, "message": "Hospital name and specialization are required."}), 400
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO hospitals (hospital_name,doctor_name,specialization,location,phone,available_days,time_slots,is_active,created_at) VALUES (?,?,?,?,?,?,?,1,?)",
              (name, doc, spec, loc, phone, days, slots, created_at))
    hid = c.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Hospital added: id={hid} name={name} spec={spec}")
    return jsonify({"success": True, "id": hid})


@app.route("/admin/doctors", methods=["POST"])
@require_admin_session
def admin_add_doctor():
    data = request.get_json()
    hid      = data.get("hospital_id")
    hname    = (data.get("hospital_name") or "").strip()
    dname    = (data.get("doctor_name") or "").strip()
    spec     = (data.get("specialization") or "").strip()
    qual     = (data.get("qualification") or "").strip()
    exp      = data.get("experience_years", 0)
    slots    = (data.get("time_slots") or "09:00 AM,10:00 AM,11:00 AM,02:00 PM,03:00 PM,04:00 PM").strip()
    days     = (data.get("available_days") or "Mon,Tue,Wed,Thu,Fri").strip()
    if not hid or not dname or not spec:
        return jsonify({"success": False, "message": "Hospital, doctor name and specialization are required."}), 400
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # fetch hospital name if not provided
    if not hname:
        row = c.execute("SELECT hospital_name FROM hospitals WHERE id=?", (hid,)).fetchone()
        hname = row[0] if row else ""
    c.execute(
        "INSERT INTO doctors (hospital_id,hospital_name,doctor_name,specialization,qualification,experience_years,time_slots,available_days,is_active,created_at) VALUES (?,?,?,?,?,?,?,?,1,?)",
        (hid, hname, dname, spec, qual, exp, slots, days, created_at)
    )
    did = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"success": True, "id": did})


@app.route("/admin/doctors", methods=["GET"])
@require_admin_session
def admin_list_doctors():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,hospital_id,hospital_name,doctor_name,specialization,qualification,experience_years,time_slots,available_days,is_active,created_at FROM doctors ORDER BY hospital_name,doctor_name")
    rows = c.fetchall()
    conn.close()
    keys = ["id","hospital_id","hospital_name","doctor_name","specialization","qualification","experience_years","time_slots","available_days","is_active","created_at"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@app.route("/admin/doctors/<int:did>", methods=["DELETE"])
@require_admin_session
def admin_delete_doctor(did):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM doctors WHERE id=?", (did,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/doctors", methods=["GET"])
def get_doctors_for_booking():
    spec = request.args.get("specialization", "").strip()
    date = request.args.get("date", "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if spec:
        c.execute("""SELECT d.id,d.doctor_name,d.specialization,d.qualification,d.experience_years,
                            d.time_slots,d.available_days,d.hospital_id,d.hospital_name,
                            h.location,h.phone
                     FROM doctors d LEFT JOIN hospitals h ON d.hospital_id=h.id
                     WHERE d.is_active=1 AND d.specialization=? ORDER BY d.doctor_name""", (spec,))
    else:
        c.execute("""SELECT d.id,d.doctor_name,d.specialization,d.qualification,d.experience_years,
                            d.time_slots,d.available_days,d.hospital_id,d.hospital_name,
                            h.location,h.phone
                     FROM doctors d LEFT JOIN hospitals h ON d.hospital_id=h.id
                     WHERE d.is_active=1 ORDER BY d.specialization,d.doctor_name""")
    rows = c.fetchall()
    keys = ["id","doctor_name","specialization","qualification","experience_years","time_slots","available_days","hospital_id","hospital_name","location","phone"]
    doctors = [dict(zip(keys, r)) for r in rows]
    if date:
        for doc in doctors:
            c.execute("SELECT preferred_time FROM appointments WHERE doctor_id=? AND preferred_date=? AND status NOT IN ('Cancelled')", (doc["id"], date))
            doc["booked_slots"] = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify(doctors)


@app.route("/admin/hospitals/<int:hid>", methods=["DELETE"])
@require_admin_session
def admin_delete_hospital(hid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM hospitals WHERE id=?", (hid,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/appointment/book", methods=["POST"])
def book_appointment():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data received."}), 400

    name      = (data.get("patient_name") or "").strip()
    age       = data.get("patient_age")
    phone     = (data.get("patient_phone") or "").strip()
    gender    = (data.get("patient_gender") or "").strip()
    blood_grp = (data.get("patient_blood_group") or "").strip()
    weight    = (data.get("patient_weight") or "").strip()
    height    = (data.get("patient_height") or "").strip()
    allergies = (data.get("patient_allergies") or "").strip()
    address   = (data.get("patient_address") or "").strip()
    email     = (data.get("patient_email") or "").strip()
    spec      = (data.get("specialization") or "").strip()
    hosp_id   = data.get("hospital_id")
    hosp_name = (data.get("hospital_name") or "").strip()
    doc_id    = data.get("doctor_id")
    doc_name  = (data.get("doctor_name") or "").strip()
    date      = (data.get("preferred_date") or "").strip()
    time_     = (data.get("preferred_time") or "").strip()
    symptoms  = (data.get("symptoms") or "").strip()

    if not name or not spec or not date or not hosp_name or not time_:
        return jsonify({"success": False, "message": "Name, specialization, hospital, date and time slot are required."}), 400

    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Duplicate check — same phone + same doctor + same date + same slot (not cancelled)
    if phone and doc_id:
        existing = c.execute(
            "SELECT id FROM appointments WHERE patient_phone=? AND doctor_id=? AND preferred_date=? AND preferred_time=? AND status NOT IN ('Cancelled')",
            (phone, doc_id, date, time_)
        ).fetchone()
        if existing:
            conn.close()
            return jsonify({"success": False, "message": f"You already have an appointment with this doctor on {date} at {time_}. Please choose a different slot."}), 409

    # Also block if same phone already has any active appointment with same doctor on same date
    if phone and doc_id:
        same_day = c.execute(
            "SELECT id FROM appointments WHERE patient_phone=? AND doctor_id=? AND preferred_date=? AND status NOT IN ('Cancelled')",
            (phone, doc_id, date)
        ).fetchone()
        if same_day:
            conn.close()
            return jsonify({"success": False, "message": f"You already have an appointment with Dr. {doc_name} on {date}. You cannot book the same doctor twice on the same day."}), 409

    c.execute(
        "INSERT INTO appointments (patient_name,patient_age,patient_phone,patient_gender,patient_blood_group,patient_weight,patient_height,patient_allergies,patient_address,patient_email,specialization,hospital_id,hospital_name,doctor_id,doctor_name,preferred_date,preferred_time,symptoms,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, age, phone, gender, blood_grp, weight, height, allergies, address, email, spec, hosp_id, hosp_name, doc_id, doc_name, date, time_, symptoms, "Pending", created_at)
    )
    appt_id = c.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Appointment booked: id={appt_id} name={name} spec={spec} hospital={hosp_name} date={date} time={time_}")
    return jsonify({"success": True, "message": "Appointment request sent! The hospital will confirm your slot shortly.", "id": appt_id})


@app.route("/appointment/list", methods=["GET"])
@require_admin_session
def list_appointments():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, patient_name, patient_age, patient_phone, specialization, hospital_name, doctor_name, preferred_date, preferred_time, symptoms, status, created_at FROM appointments ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    keys = ["id","patient_name","patient_age","patient_phone","specialization","hospital_name","doctor_name","preferred_date","preferred_time","symptoms","status","created_at"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@app.route("/appointment/<int:appt_id>/status", methods=["PATCH"])
@require_admin_session
def update_appointment_status(appt_id):
    data = request.get_json()
    new_status = (data.get("status") or "").strip()
    if new_status not in ("Pending", "Confirmed", "Cancelled", "Completed"):
        return jsonify({"success": False, "message": "Invalid status."}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE appointments SET status=? WHERE id=?", (new_status, appt_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/appointment/<int:appt_id>/note", methods=["PATCH"])
@require_admin_session
def update_appointment_note(appt_id):
    data = request.get_json()
    note = (data.get("note") or "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE appointments SET doctor_note=? WHERE id=?", (note, appt_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})



@app.route("/patient/appointments/count", methods=["GET"])
def patient_appointment_count():
    patient_id = session.get("patient_id")
    if not patient_id:
        return jsonify({"count": 0})
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pat = c.execute("SELECT phone FROM patients WHERE id=?", (patient_id,)).fetchone()
    if not pat:
        conn.close()
        return jsonify({"count": 0})
    count = c.execute(
        "SELECT COUNT(*) FROM appointments WHERE patient_phone=? AND status IN ('Pending','Confirmed')",
        (pat[0],)
    ).fetchone()[0]
    conn.close()
    return jsonify({"count": count})


@app.route("/appointment/my", methods=["GET"])
def get_my_appointments():
    # Session-based if logged in, else phone param
    patient_id = session.get("patient_id")
    if patient_id:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        pat = c.execute("SELECT phone FROM patients WHERE id=?", (patient_id,)).fetchone()
        phone = pat[0] if pat else None
    else:
        phone = request.args.get("phone", "").strip()

    if not phone:
        return jsonify({"success": False, "message": "Phone number required."}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id,patient_name,patient_age,specialization,hospital_name,doctor_name,preferred_date,preferred_time,symptoms,status,created_at,doctor_note FROM appointments WHERE patient_phone=? ORDER BY id DESC",
        (phone,)
    )
    rows = c.fetchall()
    conn.close()
    keys = ["id","patient_name","patient_age","specialization","hospital_name","doctor_name","preferred_date","preferred_time","symptoms","status","created_at","doctor_note"]
    return jsonify({"success": True, "appointments": [dict(zip(keys, r)) for r in rows]})


# ─── PATIENT AUTH ROUTES ──────────────────────────────────────────────────────

@app.route("/patient/register", methods=["POST"])
def patient_register():
    data  = request.get_json()
    name  = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    email = (data.get("email") or "").strip()
    pwd   = (data.get("password") or "").strip()
    if not name or not phone or not pwd:
        return jsonify({"success": False, "message": "Name, phone and password are required."}), 400
    if len(pwd) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO patients (name,phone,email,password_hash,created_at) VALUES (?,?,?,?,?)",
            (name, phone, email, generate_password_hash(pwd), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        pid = c.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "This phone number is already registered. Please log in."}), 409
    conn.close()
    session["patient_id"]   = pid
    session["patient_name"] = name
    session["patient_phone"] = phone
    return jsonify({"success": True, "name": name, "phone": phone})


@app.route("/patient/login", methods=["POST"])
def patient_login():
    data  = request.get_json()
    phone = (data.get("phone") or "").strip()
    pwd   = (data.get("password") or "").strip()
    if not phone or not pwd:
        return jsonify({"success": False, "message": "Phone and password are required."}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("SELECT id,name,password_hash FROM patients WHERE phone=?", (phone,)).fetchone()
    conn.close()
    if not row or not check_password_hash(row[2], pwd):
        return jsonify({"success": False, "message": "Incorrect phone number or password."}), 401
    session["patient_id"]    = row[0]
    session["patient_name"]  = row[1]
    session["patient_phone"] = phone
    return jsonify({"success": True, "name": row[1], "phone": phone})


@app.route("/patient/logout", methods=["POST"])
def patient_logout():
    session.pop("patient_id", None)
    session.pop("patient_name", None)
    session.pop("patient_phone", None)
    return jsonify({"success": True})


@app.route("/patient/session", methods=["GET"])
def patient_session():
    pid = session.get("patient_id")
    if not pid:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "name": session.get("patient_name"), "phone": session.get("patient_phone")})


@app.route("/appointment/<int:appt_id>/cancel", methods=["PATCH"])
def patient_cancel_appointment(appt_id):
    pid = session.get("patient_id")
    if not pid:
        return jsonify({"success": False, "message": "Not logged in."}), 401
    phone = session.get("patient_phone")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("SELECT status, patient_phone FROM appointments WHERE id=?", (appt_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Appointment not found."}), 404
    if row[1] != phone:
        conn.close()
        return jsonify({"success": False, "message": "Not your appointment."}), 403
    if row[0] not in ("Pending", "Confirmed"):
        conn.close()
        return jsonify({"success": False, "message": f"Cannot cancel a {row[0]} appointment."}), 400
    c.execute("UPDATE appointments SET status='Cancelled' WHERE id=?", (appt_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ─── ADMIN OVERVIEW ROUTE ────────────────────────────────────────────────────

@app.route("/admin/overview", methods=["GET"])
@require_admin_session
def admin_overview():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()

    # Appointment counts
    total_appts   = c.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    today_appts   = c.execute("SELECT COUNT(*) FROM appointments WHERE preferred_date=?", (today,)).fetchone()[0]
    pending_appts = c.execute("SELECT COUNT(*) FROM appointments WHERE status='Pending'").fetchone()[0]
    confirmed_appts = c.execute("SELECT COUNT(*) FROM appointments WHERE status='Confirmed'").fetchone()[0]

    # Hospital & doctor counts
    total_hospitals = c.execute("SELECT COUNT(*) FROM hospitals WHERE is_active=1").fetchone()[0]
    total_doctors   = c.execute("SELECT COUNT(*) FROM doctors WHERE is_active=1").fetchone()[0]

    # Blood request counts
    total_blood   = c.execute("SELECT COUNT(*) FROM blood_requests").fetchone()[0]
    pending_blood = c.execute("SELECT COUNT(*) FROM blood_requests WHERE status='Pending'").fetchone()[0]

    # Most booked specialization
    top_spec_row = c.execute(
        "SELECT specialization, COUNT(*) as cnt FROM appointments GROUP BY specialization ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    top_spec = {"name": top_spec_row[0], "count": top_spec_row[1]} if top_spec_row else {"name": "—", "count": 0}

    # Analytics: appointments per specialization
    spec_rows = c.execute(
        "SELECT specialization, COUNT(*) as cnt FROM appointments GROUP BY specialization ORDER BY cnt DESC"
    ).fetchall()
    spec_chart = [{"label": r[0], "count": r[1]} for r in spec_rows]

    # Analytics: status breakdown
    status_rows = c.execute(
        "SELECT status, COUNT(*) as cnt FROM appointments GROUP BY status"
    ).fetchall()
    status_chart = {r[0]: r[1] for r in status_rows}

    # Recent 5 appointments
    c.execute(
        "SELECT id,patient_name,doctor_name,specialization,preferred_date,preferred_time,status FROM appointments ORDER BY id DESC LIMIT 5"
    )
    recent_appts = [dict(zip(["id","patient_name","doctor_name","specialization","preferred_date","preferred_time","status"], r)) for r in c.fetchall()]

    # Recent 5 blood requests
    c.execute(
        "SELECT id,patient_name,hospital_name,blood_group,units_needed,urgency,status FROM blood_requests ORDER BY id DESC LIMIT 5"
    )
    recent_blood = [dict(zip(["id","patient_name","hospital_name","blood_group","units_needed","urgency","status"], r)) for r in c.fetchall()]

    conn.close()
    return jsonify({
        "appointments": {
            "total": total_appts, "today": today_appts,
            "pending": pending_appts, "confirmed": confirmed_appts
        },
        "hospitals": total_hospitals,
        "doctors":   total_doctors,
        "blood":     {"total": total_blood, "pending": pending_blood},
        "top_specialization": top_spec,
        "spec_chart":   spec_chart,
        "status_chart": status_chart,
        "recent_appointments": recent_appts,
        "recent_blood_requests": recent_blood
    })


# ─── BLOOD BANK ROUTES ────────────────────────────────────────────────────────

@app.route("/bloodbank/nearby", methods=["GET"])
def bloodbank_nearby():
    location = request.args.get("location", "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if location:
        c.execute(
            "SELECT id, hospital_name, location, phone FROM hospitals WHERE is_active=1 AND location LIKE ? ORDER BY hospital_name",
            (f"%{location}%",)
        )
    else:
        c.execute("SELECT id, hospital_name, location, phone FROM hospitals WHERE is_active=1 ORDER BY hospital_name")
    rows = c.fetchall()
    conn.close()
    keys = ["id", "hospital_name", "location", "phone"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@app.route("/bloodbank/request", methods=["POST"])
def submit_blood_request():
    name      = (request.form.get("patient_name") or "").strip()
    age       = request.form.get("patient_age")
    phone     = (request.form.get("patient_phone") or "").strip()
    hospital  = (request.form.get("hospital_name") or "").strip()
    blood_grp = (request.form.get("blood_group") or "").strip()
    units     = request.form.get("units_needed", 1)
    urgency   = (request.form.get("urgency") or "Normal").strip()

    if not name or not hospital or not blood_grp:
        return jsonify({"success": False, "message": "Name, hospital and blood group are required."}), 400

    prescription_filename = None
    file = request.files.get("prescription")
    if file and file.filename:
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in PRESCRIPTION_ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "message": "Prescription must be PDF, PNG or JPG."}), 400
        safe_name = secure_filename(file.filename)
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        prescription_filename = f"{ts}_{safe_name}"
        file.save(os.path.join(PRESCRIPTION_FOLDER, prescription_filename))

    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO blood_requests (patient_name, patient_age, patient_phone, hospital_name, blood_group, units_needed, urgency, prescription_filename, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (name, age, phone, hospital, blood_grp, units, urgency, prescription_filename, "Pending", created_at)
    )
    req_id = c.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Blood request submitted: id={req_id} name={name} group={blood_grp} units={units} urgency={urgency}")
    return jsonify({"success": True, "message": "Blood request submitted successfully!", "id": req_id})


@app.route("/bloodbank/requests", methods=["GET"])
@require_admin_session
def list_blood_requests():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, patient_name, patient_age, patient_phone, hospital_name, blood_group, units_needed, urgency, prescription_filename, status, created_at FROM blood_requests ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    keys = ["id","patient_name","patient_age","patient_phone","hospital_name","blood_group","units_needed","urgency","prescription_filename","status","created_at"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@app.route("/bloodbank/<int:req_id>/status", methods=["PATCH"])
@require_admin_session
def update_blood_request_status(req_id):
    data = request.get_json()
    new_status = (data.get("status") or "").strip()
    if new_status not in ("Pending", "Fulfilled", "Rejected"):
        return jsonify({"success": False, "message": "Invalid status."}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE blood_requests SET status=? WHERE id=?", (new_status, req_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ─── BLOOD DONOR ROUTES ───────────────────────────────────────────────────────

@app.route("/bloodbank/donor/register", methods=["POST"])
def register_blood_donor():
    data  = request.get_json()
    name  = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    group = (data.get("blood_group") or "").strip()
    city  = (data.get("city") or "").strip()
    if not name or not phone or not group or not city:
        return jsonify({"success": False, "message": "All fields are required."}), 400
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO blood_donors (name,phone,blood_group,city,is_available,created_at) VALUES (?,?,?,?,1,?)",
        (name, phone, group, city, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": f"Thank you {name}! You are registered as a {group} donor in {city}."})


@app.route("/admin/bloodbank/donors", methods=["GET"])
@require_admin_session
def admin_list_donors():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    group = request.args.get("blood_group", "").strip()
    city  = request.args.get("city", "").strip()
    query = "SELECT id,name,phone,blood_group,city,is_available,created_at FROM blood_donors WHERE 1=1"
    params = []
    if group: query += " AND blood_group=?";  params.append(group)
    if city:  query += " AND city LIKE ?";    params.append(f"%{city}%")
    query += " ORDER BY id DESC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    keys = ["id","name","phone","blood_group","city","is_available","created_at"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@app.route("/admin/bloodbank/donors/<int:did>/toggle", methods=["PATCH"])
@require_admin_session
def toggle_donor_availability(did):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("UPDATE blood_donors SET is_available = CASE WHEN is_available=1 THEN 0 ELSE 1 END WHERE id=?", (did,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ─── STT ROUTE ────────────────────────────────────────────────────────────────

@app.route("/api/stt", methods=["POST"])
def sarvam_speech_to_text():
    if not SARVAM_API_KEY or SARVAM_API_KEY == "your_sarvam_api_key_here":
        return jsonify({"error": "Sarvam API key not configured."}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received."}), 400

    audio_file = request.files["audio"]
    lang = request.form.get("language_code", "en-IN")

    audio_bytes = audio_file.read()
    content_type = audio_file.content_type or "audio/webm"
    orig_name = audio_file.filename or "recording.webm"
    filename = orig_name  # keep original extension so Sarvam detects format correctly

    try:
        sarvam_resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": (filename, audio_bytes, content_type)},
            data={"language_code": lang, "model": "saaras:v3"},
            timeout=30
        )
        logger.info("Sarvam STT status=%s body=%s", sarvam_resp.status_code, sarvam_resp.text[:500])
        result = sarvam_resp.json()
        # Sarvam may return 'transcript' or 'text'
        transcript = result.get("transcript") or result.get("text") or ""
        if transcript:
            return jsonify({"transcript": transcript, "language_code": result.get("language_code", lang)})
        else:
            logger.warning("Sarvam STT no transcript: %s", result)
            raw_err = result.get("message") or result.get("error") or result.get("detail") or result
            # flatten nested objects to a readable string
            if isinstance(raw_err, dict):
                err_msg = raw_err.get("message") or raw_err.get("detail") or str(raw_err)
            else:
                err_msg = str(raw_err)
            return jsonify({"error": err_msg}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": "Sarvam API timed out. Try again."}), 504
    except Exception as ex:
        logger.error("Sarvam STT error: %s", ex)
        return jsonify({"error": str(ex)}), 500

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CuraBot server starting on http://0.0.0.0:8080")
    logger.info("Log file: %s", _log_file)
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)