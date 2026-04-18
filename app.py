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

# --- UTILS ---
def extract_base_repo_url(content: str) -> str | None:
    """Ekstrak base repo URL dari berbagai format HTML murid.
    
    Menangani:
    - URL di dalam href: <a href="https://github.com/user/repo">
    - URL plain text: https://github.com/user/repo
    - Suffix .git: https://github.com/user/repo.git
    - Path spesifik: /blob/main/file.py, /tree/main, /commit/..., /pull/...
    """
    # Coba ambil dari href dulu (lebih akurat)
    href_match = re.search(r'href=["\']?(https://github\.com/[\w.-]+/[\w.-]+[^"\'>\s]*)', content)
    if href_match:
        url = href_match.group(1)
    else:
        # Fallback ke plain text URL
        text_match = re.search(r'(https://github\.com/[\w.-]+/[\w.-]+[^\s<"\']*)', content)
        if text_match:
            url = text_match.group(1)
        else:
            return None
    
    # Bersihkan URL: hapus trailing slash, .git, dan path spesifik
    url = url.rstrip('/')
    url = re.sub(r'\.git$', '', url)
    url = re.sub(r'/(blob|tree|commit|pull|issues|actions|releases)/.*', '', url)
    url = url.rstrip('/')
    
    return url

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

def fetch_all_py_files(repo_url: str, log_cb=None) -> Dict[str, str]:
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
        
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                # Skip venv and system folders entirely to save GitHub API calls
                if file_content.name.lower() in ignore_dirs:
                    log(f"    ⏭ Ignored dir: {file_content.name}/")
                    continue
                contents.extend(repo.get_contents(file_content.path))
            elif file_content.name.endswith(".py") and file_content.size < 50000:
                if 'tugas' in file_content.name.lower():
                    collected_code[file_content.name] = file_content.decoded_content.decode()
                    log(f"    ✓ Found: {file_content.name} ({file_content.size} bytes)")
                else:
                    log(f"    ⏭ Ignored: {file_content.name} (no 'tugas' in filename)")
        
        if not collected_code:
            log(f"    ⚠ No .py files found in repo!")
        else:
            log(f"  📄 Total files: {len(collected_code)}")
    except Exception as e:
        log(f"  ❌ GitHub Fetch Error: {e}")
    return collected_code

def evaluate_with_groq(student_name: str, task_id: str, codes: Dict[str, str], log_cb=None) -> Dict[str, Any]:
    def log(m):
        print(m)
        if log_cb: log_cb(m)
        
    if not codes:
        log(f"  ⚠ No code to evaluate for {student_name}")
        return {
            "score": 0, 
            "feedback": "Tidak ada file Python (.py) yang ditemukan di repository.",
            "raw_ai_response": "",
            "criteria": {}
        }
    log(f"  🤖 Sending {len(codes)} file(s) to Groq AI...")

    formatted_code = "\n".join([f"--- FILE: {name} ---\n{content}" for name, content in codes.items()])
    prompt = f"""Kamu adalah AI Grader yang menilai tugas koding Python.
Nilai tugas '{task_id}' dari murid bernama '{student_name}'.

Kode yang dikumpulkan:
{formatted_code}

Berikan penilaian VERBOSE dengan format ketat berikut (JANGAN tambahkan teks di luar format):

CORRECTNESS: [0-25] - Apakah kode berjalan benar dan menghasilkan output yang diharapkan?
CODE_QUALITY: [0-25] - Apakah kode rapi, readable, dan well-structured?
COMPLETENESS: [0-25] - Apakah semua requirement tugas terpenuhi?
BEST_PRACTICES: [0-25] - Apakah menggunakan best practices Python (naming, comments, error handling)?

TOTAL_SCORE: [0-100]

DETAIL_CORRECTNESS: [1-2 kalimat penjelasan skor correctness]
DETAIL_CODE_QUALITY: [1-2 kalimat penjelasan skor code quality]
DETAIL_COMPLETENESS: [1-2 kalimat penjelasan skor completeness]
DETAIL_BEST_PRACTICES: [1-2 kalimat penjelasan skor best practices]

FEEDBACK: [Ringkasan keseluruhan dalam 2-3 kalimat]

PENTING: Jadilah penilai yang sangat murah hati (generous). Jika kode berjalan dengan wajar dan memenuhi ekspektasi dasar, berikan skor total di kisaran 85 hingga 95. Jangan ragu memberikan 90+ jika cukup rapi. Hanya berikan nilai rendah jika kodenya benar-benar rusak parah, kosong, atau tidak relevan sama sekali dan itupun, jangan berikan 0 tapi 65, itu adalah aturan perusahaanku soal nilai terendah, dan tertinggi adalah 95 Silakan tetap objektif tapi generous, 90+ kalau bagus tapi jangan semua orang dikasih 95 juga, harus tetap objektif ya!
"""
    
    try:
        completion = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        result = completion.choices[0].message.content
        log(f"  ✅ AI response received!")
        
        # Parse skor per kriteria
        def parse_int(pattern, text):
            m = re.search(pattern, text)
            return int(m.group(1)) if m else 0
        
        def parse_str(pattern, text):
            m = re.search(pattern, text)
            return m.group(1).strip() if m else ""
        
        correctness = parse_int(r'CORRECTNESS:\s*(\d+)', result)
        code_quality = parse_int(r'CODE_QUALITY:\s*(\d+)', result)
        completeness = parse_int(r'COMPLETENESS:\s*(\d+)', result)
        best_practices = parse_int(r'BEST_PRACTICES:\s*(\d+)', result)
        total = parse_int(r'TOTAL_SCORE:\s*(\d+)', result)
        
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
            "criteria": criteria
        }
    except Exception as e:
        log(f"  ❌ Groq Error: {e}")
        return {"score": 0, "feedback": f"Error Groq: {str(e)}", "raw_ai_response": "", "criteria": {}}

# --- ENDPOINTS (Sesuai Struktur 4 Tab) ---

# TAB 1: DASHBOARD
@app.get("/dashboard-stats")
async def get_dashboard_stats():
    """Statistik dasar untuk halaman depan."""
    total_students = await db.students.count_documents({})
    # Bisa ditambahkan agregasi MongoDB lain untuk hitung rata-rata nilai
    return {
        "total_students": total_students,
        "active_database": DB_NAME,
        "llm_engine": "Groq Llama-3.3-70b"
    }

# TAB 2: MURID
@app.post("/upload-students")
async def upload_students(file: UploadFile = File(...)):
    """Upload CSV dan simpan ke koleksi 'students'."""
    df = pd.read_csv(file.file)
    if 'nama' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV harus punya kolom 'nama'")
    
    students_data = [{"nama": row['nama']} for _, row in df.iterrows()]
    await db.students.drop() 
    await db.students.insert_many(students_data)
    
    # Buat dokumen kosong di koleksi grades untuk tiap murid agar siap diisi Tugas 1-17
    await db.grades.drop()
    grade_docs = [{"nama": row['nama'], "tasks": {}} for _, row in df.iterrows()]
    await db.grades.insert_many(grade_docs)
    
    return {"message": f"Berhasil mengimpor {len(students_data)} murid"}

@app.get("/students")
async def get_students():
    """Ambil daftar semua murid."""
    students = await db.students.find({}, {"_id": 0}).to_list(length=1000)
    return students

# TAB 3: NILAI KESELURUHAN
@app.get("/grades")
async def get_all_grades():
    """Mengembalikan daftar murid beserta nilai lengkap Tugas 1 - 17."""
    grades = await db.grades.find({}, {"_id": 0}).to_list(length=1000)
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
    students_db = await db.students.find().to_list(length=1000)
    grades_db = await db.grades.find().to_list(length=1000)
    grades_map = {g['nama']: g.get('tasks', {}).get(task_id) for g in grades_db}
    
    report = []
    for s in students_db:
        matched_folder = match_name(s['nama'], all_folders)
        github_url = None
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
                    github_url = extract_base_repo_url(content)
        
        report.append(StudentStatus(
            student_name_csv=s['nama'],
            folder_name=matched_folder,
            github_url=github_url,
            has_html=has_html,
            html_content=html_content,
            is_submitted=True if github_url else False,
            grade_result=grade_result
        ))
    return report

@app.get("/grade-student-stream/{task_id}/{student_name}")
async def grade_student_stream(task_id: str, student_name: str):
    """Menjalankan proses AI Grader via SSE (Server-Sent Events) untuk log realtime."""
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def sync_worker():
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
                github_url = extract_base_repo_url(f.read())
            
            if not github_url:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": "URL GitHub tidak ditemukan dalam file HTML"})
                return
            
            qlog(f"🔗 GitHub URL: {github_url}")

            # Ambil kode & Nilai pakai AI dengan callback
            codes = fetch_all_py_files(github_url, log_cb=qlog)
            ai_result = evaluate_with_groq(student_name, task_id, codes, log_cb=qlog)
            
            db_payload = {
                "score": ai_result["score"],
                "feedback": ai_result["feedback"],
                "github_url": github_url,
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
    threading.Thread(target=sync_worker).start()
    
    async def event_generator():
        while True:
            item = await queue.get()
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
        github_url = extract_base_repo_url(f.read())
    codes = fetch_all_py_files(github_url)
    ai_result = evaluate_with_groq(student_name, task_id, codes)
    db_payload = {
        "score": ai_result["score"], "feedback": ai_result["feedback"], "github_url": github_url,
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
                github_url = extract_base_repo_url(f.read())
            
            if not github_url:
                print(f"  ❌ URL GitHub tidak ditemukan")
                results.append({"nama": student_name, "task": task_id, "status": "error", "error": "URL GitHub tidak ditemukan", "result": None})
                continue
            
            print(f"  🔗 {github_url}")
            codes = fetch_all_py_files(github_url)
            ai_result = evaluate_with_groq(student_name, task_id, codes)
            
            db_payload = {
                "score": ai_result["score"],
                "feedback": ai_result["feedback"],
                "github_url": github_url,
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