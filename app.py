import os
import re
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from github import Github
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import json
import asyncio
import threading
from fastapi.responses import StreamingResponse
from grader_utils import trim_python_code, run_static_analysis, estimate_tokens, chunk_python_code

load_dotenv()

app = FastAPI(title="AI Grader Pro - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi
GITHUB_PAT = os.getenv("GITHUB_PAT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
# BASE_DIR sekarang menunjuk ke root folder penampung semua tugas
BASE_DIR = "TugasMurid" 

g = Github(GITHUB_PAT)
client_groq = Groq(api_key=GROQ_API_KEY)
db_client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
db = db_client[DB_NAME]

# --- SCHEMAS ---
class StudentStatus(BaseModel):
    student_name_csv: str
    folder_name: Optional[str] = None
    github_url: Optional[str] = None
    task_type: Optional[str] = None # 'github' or 'colab'
    has_html: bool = False
    html_content: Optional[str] = None # Untuk nampilin isi file di Frontend
    is_submitted: bool = False
    grade_result: Optional[Dict[str, Any]] = None

class GradeUpdate(BaseModel):
    score: int
    feedback: str

class BatchGradeRequest(BaseModel):
    task_id: str
    student_names: List[str]  # Maksimal 5
    force_ext: Optional[str] = None

class EngineSettingsUpdate(BaseModel):
    engine: str
    groq_keys: List[str]
    ollama_url: str
    ollama_model: str

async def get_engine_settings():
    settings = await db.settings.find_one({"_id": "engine_config"})
    if not settings:
        default_key = os.getenv("GROQ_API_KEY", "")
        settings = {
            "_id": "engine_config",
            "engine": "groq",
            "groq_keys": [default_key] if default_key else [],
            "active_groq_index": 0,
            "ollama_url": "http://127.0.0.1:11434/api/generate",
            "ollama_model": "llama3"
        }
        await db.settings.update_one({"_id": "engine_config"}, {"$set": settings}, upsert=True)
    return settings

# --- UTILS ---
def extract_task_url_and_type(content: str) -> tuple[str | None, str | None]:
    """Ekstrak URL tugas dan tipenya (github atau colab)."""
    # Cek Colab dulu
    colab_match = re.search(r'(https://colab\.research\.google\.com/drive/[\w-]+)', content)
    if colab_match:
        return colab_match.group(1), "colab"
        
    # Cek GitHub
    href_match = re.search(r'href=["\']?(https://github\.com/[\w.-]+/[\w.-]+[^"\'>\s]*)', content)
    if href_match:
        url = href_match.group(1)
    else:
        text_match = re.search(r'(https://github\.com/[\w.-]+/[\w.-]+[^\s<"\']*)', content)
        if text_match:
            url = text_match.group(1)
        else:
            return None, None
            
    url = url.rstrip('/')
    url = re.sub(r'\.git$', '', url)
    url = re.sub(r'/(blob|tree|commit|pull|issues|actions|releases)/.*', '', url)
    url = url.rstrip('/')
    
    return url, "github"

def match_name(student_name: str, folder_list: List[str]) -> str | None:
    """Cocokkan nama murid dari CSV dengan folder Moodle.
    
    Format folder Moodle: 'Nama Lengkap_12345_assignsubmission_onlinetext'
    Strategi matching (berurutan):
    1. Ekstrak nama dari folder (sebelum _ID_) lalu bandingkan persis
    2. Cek apakah nama CSV terkandung di awal folder (sebelum _ID)
    3. Cek apakah nama CSV terkandung di folder (case insensitive)
    """
    clean_name = student_name.lower().strip()
    
    for folder in folder_list:
        # Ekstrak nama dari folder: ambil bagian sebelum _ANGKA_
        folder_name_part = re.split(r'_\d+_', folder)[0].lower().strip()
        
        # Strategy 1: Persis sama
        if clean_name == folder_name_part:
            return folder
        
        # Strategy 2: Folder Moodle kadang duplikat nama (e.g. "Hendry Hendry")
        # Cek apakah nama CSV ada di folder_name_part
        if clean_name in folder_name_part or folder_name_part.startswith(clean_name):
            return folder
    
    # Strategy 3: Fallback - cek semua kata nama ada di folder
    for folder in folder_list:
        folder_lower = folder.lower()
        name_parts = clean_name.split()
        if len(name_parts) >= 2 and all(part in folder_lower for part in name_parts):
            return folder
    
    return None

import requests

def fetch_colab_notebook(colab_url: str, log_cb=None) -> Dict[str, str]:
    def log(m):
        print(m)
        if log_cb: log_cb(m)
        
    try:
        match = re.search(r'drive/([\w-]+)', colab_url)
        if not match:
            log("  ❌ Invalid Colab URL")
            return {}
        file_id = match.group(1)
        log(f"  📂 Fetching Colab ID: {file_id}")
        
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        
        if response.status_code == 200:
            if "<!doctype html>" in response.text.lower() or "accounts.google.com" in response.text:
                log("    ⚠ Akses Ditolak: Link Colab di-Private. Murid harus mengubah Share ke 'Anyone with the link'")
                return {}
                
            from grader_utils import process_ipynb_content
            raw_json = response.text
            processed_code = process_ipynb_content(raw_json)
            if processed_code:
                log(f"    ✓ Successfully processed Colab Notebook")
                return {f"colab_{file_id}.ipynb": processed_code}
            else:
                log("    ⚠ Notebook is empty or invalid JSON")
        else:
            log(f"    ❌ Failed to download, status: {response.status_code}")
    except Exception as e:
        log(f"  ❌ Colab Fetch Error: {e}")
    return {}

def fetch_all_py_files(repo_url: str, task_id: str, force_ext: Optional[str] = None, log_cb=None) -> Dict[str, str]:
    def log(m):
        print(m)
        if log_cb: log_cb(m)
        
    collected_code = {}
    try:
        repo_name = repo_url.replace("https://github.com/", "").strip("/")
        log(f"  📂 Fetching repo: {repo_name}")
        repo = g.get_repo(repo_name)
        contents = repo.get_contents("")
        
        ignore_dirs = {'.venv', 'venv', 'env', '.env', '__pycache__', 'node_modules', '.git'}
        
        # Buat variasi nama tugas, misal "tugas-5" -> ["tugas-5", "tugas5", "tugas_5"]
        base_num = re.search(r'\d+', task_id)
        if base_num:
            num = base_num.group(0)
            valid_keywords = [task_id.lower(), f"tugas{num}", f"tugas_{num}"]
        else:
            valid_keywords = [task_id.lower()]

        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                # Skip venv and system folders entirely to save GitHub API calls
                if file_content.name.lower() in ignore_dirs:
                    log(f"    ⏭ Ignored dir: {file_content.name}/")
                    continue
                contents.extend(repo.get_contents(file_content.path))
            elif (file_content.name.endswith(".py") or file_content.name.endswith(".ipynb")) and file_content.size < 5000000:
                if force_ext == 'py' and not file_content.name.endswith('.py'):
                    continue
                if force_ext == 'ipynb' and not file_content.name.endswith('.ipynb'):
                    continue
                
                name_lower = file_content.name.lower()
                if any(kw in name_lower for kw in valid_keywords):
                    try:
                        if file_content.content is None and file_content.download_url:
                            resp = requests.get(file_content.download_url)
                            if resp.status_code == 200:
                                content_str = resp.text
                            else:
                                log(f"    ⚠ Failed to download large file {file_content.name}")
                                continue
                        else:
                            content_str = file_content.decoded_content.decode('utf-8')
                    except Exception as e:
                        log(f"    ⚠ Could not decode {file_content.name}: {e}")
                        continue
                        
                    if file_content.name.endswith(".ipynb"):
                        from grader_utils import process_ipynb_content
                        content_str = process_ipynb_content(content_str)
                    collected_code[file_content.name] = content_str
                    log(f"    ✓ Found: {file_content.name} ({file_content.size} bytes)")
                else:
                    log(f"    ⏭ Ignored: {file_content.name} (bukan untuk tugas ini)")
        
        if not collected_code:
            log(f"    ⚠ No .py/.ipynb files found in repo!")
        else:
            log(f"  📄 Total files: {len(collected_code)}")
    except Exception as e:
        log(f"  ❌ GitHub Fetch Error: {e}")
    return collected_code

def run_groq_inference(prompt: str, groq_keys: List[str], start_index: int, log) -> tuple[Optional[str], int]:
    current_index = start_index
    attempts = 0
    while attempts < len(groq_keys):
        key = groq_keys[current_index]
        if not key.strip():
            current_index = (current_index + 1) % len(groq_keys)
            attempts += 1
            continue
            
        client = Groq(api_key=key.strip())
        try:
            log(f"  🤖 Trying Groq Key [{current_index}]...")
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            return completion.choices[0].message.content, current_index
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "413" in err_str or "rate limit" in err_str or "tokens" in err_str:
                log(f"  ⚠ Groq Key [{current_index}] Rate Limited/Error. Switching key...")
                current_index = (current_index + 1) % len(groq_keys)
                attempts += 1
            else:
                log(f"  ❌ Groq Error: {e}")
                return None, current_index
    log("  ❌ All Groq keys exhausted or invalid.")
    return None, current_index

def run_ollama_inference(prompt: str, ollama_url: str, ollama_model: str, log) -> Optional[str]:
    log(f"  🤖 Sending to Ollama ({ollama_model}) at {ollama_url}...")
    try:
        response = requests.post(
            ollama_url,
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1000
                }
            },
            timeout=180
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "")
        else:
            log(f"  ❌ Ollama Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log(f"  ❌ Ollama Connection Error: {e}")
        return None

def evaluate_task(student_name: str, task_id: str, codes: Dict[str, str], engine_settings: dict, log_cb=None) -> Dict[str, Any]:
    def log(m):
        print(m)
        if log_cb: log_cb(m)
        
    if not codes:
        log(f"  ⚠ No code to evaluate for {student_name}")
        return {
            "score": 0, 
            "feedback": "Tidak ada file yang ditemukan atau valid di repository.",
            "raw_ai_response": "",
            "criteria": {},
            "new_groq_index": engine_settings.get("active_groq_index", 0)
        }
    log(f"  🤖 Sending {len(codes)} file(s) to AI Engine ({engine_settings.get('engine', 'groq')})...")

    processed_codes = {}
    static_analysis_reports = {}
    total_tokens = 0
    
    # Phase 1: Preprocessing & Static Analysis
    for name, content in codes.items():
        log(f"    ✂️ Trimming tokens for {name}...")
        trimmed = trim_python_code(content)
        
        log(f"    🔍 Running Static Analysis (flake8 & radon) on {name}...")
        analysis = run_static_analysis(trimmed)
        static_analysis_reports[name] = analysis
        
        processed_codes[name] = trimmed
        total_tokens += estimate_tokens(trimmed)
        
    log(f"    📈 Total Estimated Tokens after trimming: {total_tokens}")

    # Build formatted code string WITH analysis
    formatted_code_parts = []
    for name, content in processed_codes.items():
        analysis = static_analysis_reports[name]
        part = f"--- FILE: {name} ---\n{content}\n\n"
        part += f"[STATIC ANALYSIS REPORT FOR {name}]\n"
        part += f"Lint Errors (Flake8):\n{analysis['lint_report']}\n"
        part += f"Complexity (Radon):\n{analysis['complexity_report']}\n"
        formatted_code_parts.append(part)
        
    formatted_code = "\n".join(formatted_code_parts)

    prompt = f"""Kamu adalah AI Grader untuk menilai tugas coding Python secara objektif namun sangat generous/friendly dalam pemberian skor.

Tugas ID: '{task_id}'
Nama Murid: '{student_name}'

Berikut kode yang dikumpulkan beserta hasil static analysis (Flake8 & Radon):
{formatted_code}

Instruksi Penilaian:
- Gunakan hasil [STATIC ANALYSIS REPORT] sebagai acuan utama untuk menilai:
  - CODE_QUALITY
  - BEST_PRACTICES
- Jangan menebak sendiri syntax error, warning, atau kompleksitas jika sudah tersedia pada laporan.
- Fokus penilaian logika program dan hasil eksekusi untuk:
  - CORRECTNESS
  - COMPLETENESS
- Asumsikan kode dijalankan pada konteks tugas normal kecuali jelas-jelas broken.

Skema Penilaian:
- CORRECTNESS (0-25):
  Apakah kode berjalan dengan benar dan menghasilkan output yang sesuai?

- CODE_QUALITY (0-25):
  Apakah kode rapi, readable, terstruktur, dan maintainable?
  Gunakan hasil Flake8 & Radon sebagai referensi utama.

- COMPLETENESS (0-25):
  Apakah seluruh requirement inti tugas telah terpenuhi?

- BEST_PRACTICES (0-25):
  Apakah sudah mengikuti praktik Python yang baik?
  Contoh: naming convention, modularity, readability, handling sederhana, dsb.

ATURAN PENTING SCORING:
- Jadilah penilai yang generous.
- Jika kode berjalan cukup baik dan memenuhi ekspektasi dasar:
  - berikan total skor sekitar 85–95.
- Jangan terlalu keras pada kekurangan minor.
- Nilai 90+ sangat diperbolehkan jika kode cukup rapi dan fungsional.
- Hanya berikan nilai rendah jika:
  - kode benar-benar rusak,
  - kosong,
  - atau tidak relevan dengan tugas.
- Bahkan pada kasus buruk sekalipun:
  - jangan memberi di bawah 65.
- Nilai maksimum adalah 95.

WAJIB gunakan format output berikut secara ketat.
JANGAN menambahkan teks apa pun di luar format ini:

CORRECTNESS: [0-25] - [penjelasan singkat]
CODE_QUALITY: [0-25] - [penjelasan singkat]
COMPLETENESS: [0-25] - [penjelasan singkat]
BEST_PRACTICES: [0-25] - [penjelasan singkat]

TOTAL_SCORE: [0-100]

DETAIL_CORRECTNESS: [1-2 kalimat]
DETAIL_CODE_QUALITY: [1-2 kalimat]
DETAIL_COMPLETENESS: [1-2 kalimat]
DETAIL_BEST_PRACTICES: [1-2 kalimat]

FEEDBACK: [Ringkasan keseluruhan dalam 2-3 kalimat]
"""
    
    try:
        result_text = None
        new_index = engine_settings.get("active_groq_index", 0)
        engine = engine_settings.get("engine", "groq")
        
        if engine == "groq" and engine_settings.get("groq_keys"):
            result_text, new_index = run_groq_inference(prompt, engine_settings["groq_keys"], new_index, log)
            
        if not result_text and engine == "groq":
            if engine_settings.get("ollama_url"):
                log("  ⚠ Fallback ke Ollama karena Groq gagal/habis...")
                result_text = run_ollama_inference(prompt, engine_settings["ollama_url"], engine_settings["ollama_model"], log)
            else:
                log("  ❌ Groq gagal dan Ollama belum dikonfigurasi.")
                
        if engine == "ollama":
            result_text = run_ollama_inference(prompt, engine_settings["ollama_url"], engine_settings["ollama_model"], log)
            
        if not result_text:
             return {
                "score": 0, "feedback": "AI Engine gagal memproses tugas ini. Cek API Key atau status Ollama.",
                "raw_ai_response": "", "criteria": {}, "new_groq_index": new_index
             }
             
        log(f"  ✅ AI response received!")
        result = result_text
        
        # Parse skor per kriteria
        def parse_int(pattern, text):
            m = re.search(pattern, text)
            return int(m.group(1)) if m else 0
        
        def parse_str(pattern, text):
            m = re.search(pattern, text)
            return m.group(1).strip() if m else ""
        
        correctness = parse_int(r'CORRECTNESS[^\d]*(\d+)', result)
        code_quality = parse_int(r'CODE_QUALITY[^\d]*(\d+)', result)
        completeness = parse_int(r'COMPLETENESS[^\d]*(\d+)', result)
        best_practices = parse_int(r'BEST_PRACTICES[^\d]*(\d+)', result)
        total = parse_int(r'TOTAL_SCORE[^\d]*(\d+)', result)
        
        # Fallback: jika total tidak di-parse, hitung manual
        if total == 0 and (correctness + code_quality + completeness + best_practices) > 0:
            total = correctness + code_quality + completeness + best_practices
        
        criteria = {
            "correctness": {
                "score": correctness,
                "max": 25,
                "detail": parse_str(r'DETAIL_CORRECTNESS:\s*(.*?)(?:\n|$)', result)
            },
            "code_quality": {
                "score": code_quality,
                "max": 25,
                "detail": parse_str(r'DETAIL_CODE_QUALITY:\s*(.*?)(?:\n|$)', result)
            },
            "completeness": {
                "score": completeness,
                "max": 25,
                "detail": parse_str(r'DETAIL_COMPLETENESS:\s*(.*?)(?:\n|$)', result)
            },
            "best_practices": {
                "score": best_practices,
                "max": 25,
                "detail": parse_str(r'DETAIL_BEST_PRACTICES:\s*(.*?)(?:\n|$)', result)
            }
        }
        
        feedback = parse_str(r'FEEDBACK:\s*(.*)', result)
        if not feedback:
            feedback = "Gagal mem-parsing feedback AI."
        
        log(f"  📊 Score: {total}/100 (C:{correctness} Q:{code_quality} CM:{completeness} BP:{best_practices})")
        return {
            "score": total,
            "feedback": feedback,
            "raw_ai_response": result,
            "criteria": criteria,
            "new_groq_index": new_index
        }
    except Exception as e:
        log(f"  ❌ AI Parse Error: {e}")
        return {"score": 0, "feedback": f"Error Parsing AI: {str(e)}", "raw_ai_response": "", "criteria": {}, "new_groq_index": new_index}

# --- ENDPOINTS (Sesuai Struktur 4 Tab) ---

# TAB 1: DASHBOARD
@app.get("/dashboard-stats")
async def get_dashboard_stats():
    """Statistik dasar untuk halaman depan."""
    total_students = await db.students.count_documents({})
    settings = await get_engine_settings()
    engine_name = f"Groq (Keys: {len(settings.get('groq_keys', []))})" if settings.get('engine') == 'groq' else f"Ollama ({settings.get('ollama_model')})"
    return {
        "total_students": total_students,
        "active_database": DB_NAME,
        "llm_engine": engine_name
    }

@app.get("/engine-settings")
async def get_engine_settings_route():
    settings = await get_engine_settings()
    # hapus id object untuk mempermudah pydantic JSON serialization
    settings.pop("_id", None)
    return settings

@app.post("/engine-settings")
async def update_engine_settings(data: EngineSettingsUpdate):
    await db.settings.update_one(
        {"_id": "engine_config"},
        {"$set": {
            "engine": data.engine,
            "groq_keys": data.groq_keys,
            "ollama_url": data.ollama_url,
            "ollama_model": data.ollama_model
        }},
        upsert=True
    )
    return {"message": "Settings updated"}

# TAB 2: MURID
@app.post("/upload-students")
async def upload_students(file: UploadFile = File(...)):
    """Upload CSV dan simpan ke koleksi 'students'."""
    df = pd.read_csv(file.file)
    if 'nama' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV harus punya kolom 'nama'")
    
    students_data = [{"nama": row['nama'], "order": idx} for idx, row in df.iterrows()]
    await db.students.drop() 
    await db.students.insert_many(students_data)
    
    # Buat dokumen kosong di koleksi grades untuk tiap murid agar siap diisi Tugas 1-17
    await db.grades.drop()
    grade_docs = [{"nama": row['nama'], "order": idx, "tasks": {}} for idx, row in df.iterrows()]
    await db.grades.insert_many(grade_docs)
    
    return {"message": f"Berhasil mengimpor {len(students_data)} murid"}

@app.get("/students")
async def get_students():
    """Ambil daftar semua murid."""
    students = await db.students.find({}, {"_id": 0}).sort("order", 1).to_list(length=1000)
    return students

# TAB 3: NILAI KESELURUHAN
@app.get("/grades")
async def get_all_grades():
    """Mengembalikan daftar murid beserta nilai lengkap Tugas 1 - 17."""
    grades = await db.grades.find({}, {"_id": 0}).sort("order", 1).to_list(length=1000)
    return grades

# TAB 4: DETEKSI & GRADING AI
@app.get("/available-tasks")
async def get_available_tasks():
    """Scan folder utama dan deteksi tugas apa saja yang foldernya tersedia."""
    if not os.path.exists(BASE_DIR):
        return []
    
    # Ambil folder seperti 'tugas-1', 'tugas-3'
    tasks = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    tasks.sort()
    return tasks

@app.get("/check-submission-status/{task_id}", response_model=List[StudentStatus])
async def check_status_for_task(task_id: str):
    """Cek siapa saja yang sudah kumpul file HTML untuk tugas tertentu."""
    task_dir = os.path.join(BASE_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail=f"Folder {task_id} tidak ditemukan")
    
    all_folders = [f for f in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, f))]
    students_db = await db.students.find().sort("order", 1).to_list(length=1000)
    grades_db = await db.grades.find().to_list(length=1000)
    grades_map = {g['nama']: g.get('tasks', {}).get(task_id) for g in grades_db}
    
    report = []
    for s in students_db:
        matched_folder = match_name(s['nama'], all_folders)
        github_url = None
        task_type = None
        has_html = False
        html_content = None
        grade_result = grades_map.get(s['nama'])
        
        if matched_folder:
            html_path = os.path.join(task_dir, matched_folder, "onlinetext.html")
            if os.path.exists(html_path):
                has_html = True
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    html_content = content # Kirim mentahan HTML agar bisa dirender Frontend
                    github_url, task_type = extract_task_url_and_type(content)
        
        report.append(StudentStatus(
            student_name_csv=s['nama'],
            folder_name=matched_folder,
            github_url=github_url,
            task_type=task_type,
            has_html=has_html,
            html_content=html_content,
            is_submitted=True if github_url else False,
            grade_result=grade_result
        ))
    return report

@app.get("/grade-student-stream/{task_id}/{student_name}")
async def grade_student_stream(task_id: str, student_name: str, force_ext: Optional[str] = None):
    """Menjalankan proses AI Grader via SSE (Server-Sent Events) untuk log realtime."""
    engine_settings = await get_engine_settings()
    
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def sync_worker(settings_dict):
        def qlog(msg):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": msg})
            
        try:
            qlog(f"🎓 GRADING: {student_name} | Task: {task_id}")
            task_dir = os.path.join(BASE_DIR, task_id)
            all_folders = os.listdir(task_dir) if os.path.exists(task_dir) else []
            matched_folder = match_name(student_name, all_folders)
            
            if not matched_folder:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": f"Folder tidak ditemukan untuk {student_name}"})
                return
            
            qlog(f"📁 Matched folder: {matched_folder}")
            html_path = os.path.join(task_dir, matched_folder, "onlinetext.html")
            if not os.path.exists(html_path):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": "File onlinetext.html tidak ditemukan"})
                return

            with open(html_path, "r", encoding="utf-8") as f:
                task_url, task_type = extract_task_url_and_type(f.read())
            
            if not task_url:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": "URL tugas (GitHub/Colab) tidak ditemukan dalam file HTML"})
                return
            
            qlog(f"🔗 Task URL ({task_type}): {task_url}")

            # Ambil kode & Nilai pakai AI dengan callback
            if task_type == 'colab':
                if force_ext == 'py':
                    qlog("    ⚠ Diabaikan karena Manual Override = Hanya .py")
                    codes = {}
                else:
                    codes = fetch_colab_notebook(task_url, log_cb=qlog)
            else:
                codes = fetch_all_py_files(task_url, task_id, force_ext=force_ext, log_cb=qlog)
                
            ai_result = evaluate_task(student_name, task_id, codes, settings_dict, log_cb=qlog)
            
            # Jika groq index berubah, update settings_dict memory
            if "new_groq_index" in ai_result and ai_result["new_groq_index"] != settings_dict.get("active_groq_index"):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "update_index", "new_index": ai_result["new_groq_index"]})
            
            db_payload = {
                "score": ai_result["score"],
                "feedback": ai_result["feedback"],
                "github_url": task_url,
                "files_analyzed": list(codes.keys()),
                "criteria": ai_result.get("criteria", {}),
                "raw_ai_response": ai_result.get("raw_ai_response", "")
            }
            
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "payload": db_payload})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

    # Start thread
    threading.Thread(target=sync_worker, args=(engine_settings,)).start()
    
    async def event_generator():
        while True:
            item = await queue.get()
            if item["type"] == "update_index":
                await db.settings.update_one({"_id": "engine_config"}, {"$set": {"active_groq_index": item["new_index"]}})
                continue
            if item["type"] == "done":
                break
            elif item["type"] == "result":
                # Save to DB async here
                await db.grades.update_one(
                    {"nama": student_name}, 
                    {"$set": {f"tasks.{task_id}": item["payload"]}}
                )
                yield f"data: {json.dumps(item)}\n\n"
            else:
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/grade-student/{task_id}/{student_name}")
async def grade_one_student(task_id: str, student_name: str):
    """(DEPRECATED) Fallback endpoint lama tanpa stream."""
    task_dir = os.path.join(BASE_DIR, task_id)
    all_folders = os.listdir(task_dir) if os.path.exists(task_dir) else []
    matched_folder = match_name(student_name, all_folders)
    if not matched_folder:
        raise HTTPException(status_code=404, detail="Folder tidak ditemukan")
    html_path = os.path.join(task_dir, matched_folder, "onlinetext.html")
    with open(html_path, "r", encoding="utf-8") as f:
        task_url, task_type = extract_task_url_and_type(f.read())
        
    if task_type == 'colab':
        if force_ext == 'py':
            codes = {}
        else:
            codes = fetch_colab_notebook(task_url)
    else:
        codes = fetch_all_py_files(task_url, task_id, force_ext=force_ext)
        
    engine_settings = await get_engine_settings()
    ai_result = evaluate_task(student_name, task_id, codes, engine_settings)
    
    if "new_groq_index" in ai_result and ai_result["new_groq_index"] != engine_settings.get("active_groq_index"):
        await db.settings.update_one({"_id": "engine_config"}, {"$set": {"active_groq_index": ai_result["new_groq_index"]}})
    db_payload = {
        "score": ai_result["score"], "feedback": ai_result["feedback"], "github_url": task_url,
        "files_analyzed": list(codes.keys()), "criteria": ai_result.get("criteria", {}), "raw_ai_response": ai_result.get("raw_ai_response", "")
    }
    await db.grades.update_one({"nama": student_name}, {"$set": {f"tasks.{task_id}": db_payload}})
    return {"nama": student_name, "task": task_id, "result": db_payload}

@app.post("/grade-batch")
async def grade_batch(request: BatchGradeRequest):
    """Grade beberapa murid sekaligus (maks 5)."""
    if len(request.student_names) > 5:
        raise HTTPException(status_code=400, detail="Maksimal 5 murid sekaligus")
    
    task_id = request.task_id
    task_dir = os.path.join(BASE_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail=f"Folder {task_id} tidak ditemukan")
    
    total = len(request.student_names)
    print(f"\n{'#'*60}")
    print(f"📦 BATCH GRADING: {total} students | Task: {task_id}")
    print(f"{'#'*60}")
    
    engine_settings = await get_engine_settings()
    all_folders = os.listdir(task_dir)
    results = []
    
    for idx, student_name in enumerate(request.student_names, 1):
        print(f"\n--- [{idx}/{total}] {student_name} ---")
        try:
            matched_folder = match_name(student_name, all_folders)
            if not matched_folder:
                print(f"  ❌ Folder tidak ditemukan")
                results.append({"nama": student_name, "task": task_id, "status": "error", "error": "Folder tidak ditemukan", "result": None})
                continue
            
            print(f"  📁 Matched: {matched_folder}")
            html_path = os.path.join(task_dir, matched_folder, "onlinetext.html")
            if not os.path.exists(html_path):
                print(f"  ❌ File HTML tidak ada")
                results.append({"nama": student_name, "task": task_id, "status": "error", "error": "File HTML tidak ditemukan", "result": None})
                continue
            
            with open(html_path, "r", encoding="utf-8") as f:
                task_url, task_type = extract_task_url_and_type(f.read())
            
            if not task_url:
                print(f"  ❌ URL tugas tidak ditemukan")
                results.append({"nama": student_name, "task": task_id, "status": "error", "error": "URL Tugas tidak ditemukan", "result": None})
                continue
            
            print(f"  🔗 {task_url} ({task_type})")
            if task_type == 'colab':
                if request.force_ext == 'py':
                    codes = {}
                else:
                    codes = fetch_colab_notebook(task_url)
            else:
                codes = fetch_all_py_files(task_url, task_id, force_ext=request.force_ext)
                
            ai_result = evaluate_task(student_name, task_id, codes, engine_settings)
            if "new_groq_index" in ai_result and ai_result["new_groq_index"] != engine_settings.get("active_groq_index"):
                engine_settings["active_groq_index"] = ai_result["new_groq_index"]
                await db.settings.update_one({"_id": "engine_config"}, {"$set": {"active_groq_index": ai_result["new_groq_index"]}})
            
            db_payload = {
                "score": ai_result["score"],
                "feedback": ai_result["feedback"],
                "github_url": task_url,
                "files_analyzed": list(codes.keys()),
                "criteria": ai_result.get("criteria", {}),
                "raw_ai_response": ai_result.get("raw_ai_response", "")
            }
            
            await db.grades.update_one(
                {"nama": student_name},
                {"$set": {f"tasks.{task_id}": db_payload}}
            )
            
            print(f"  ✅ DONE → Score: {ai_result['score']}/100")
            results.append({"nama": student_name, "task": task_id, "status": "success", "error": None, "result": db_payload})
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({"nama": student_name, "task": task_id, "status": "error", "error": str(e), "result": None})
    
    print(f"\n{'#'*60}")
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"📦 BATCH COMPLETE: {success_count}/{total} berhasil")
    print(f"{'#'*60}\n")
    
    return {"batch_results": results}