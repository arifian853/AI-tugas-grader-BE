import ast
import re
import subprocess
import tempfile
import os
import tiktoken
import json

def estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Mengestimasi jumlah token dalam sebuah teks."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def trim_python_code(code: str) -> str:
    """
    Minifikasi file python untuk menghemat token Groq:
    - Menghapus baris kosong
    - Menghapus komentar satu baris (#)
    - Docstrings tetap dipertahankan karena sering dinilai.
    """
    lines = code.split('\n')
    trimmed_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip baris kosong
        if not stripped:
            continue
        # Skip full-line comments
        if stripped.startswith('#'):
            continue
        # Menghapus inline comments dengan regex sederhana (tidak 100% sempurna untuk string yang mengandung # tapi cukup aman)
        if '#' in line:
            line = re.sub(r'(?<!["\'])\s*#.*', '', line)
            
        trimmed_lines.append(line)
    return '\n'.join(trimmed_lines)

def run_static_analysis(code: str) -> dict:
    """
    Menjalankan flake8 dan radon pada kode untuk mendapatkan laporan statis.
    Ini menggantikan sebagian tugas LLM agar LLM fokus ke logika.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # 1. FLAKE8 (Syntax, Style, Error)
        # Menangkap error (E), fatal (F), warning (W). Baris maksimal 120.
        flake8_result = subprocess.run(
            ['flake8', temp_file_path, '--max-line-length=120', '--select=E,F,W', '--format=%(row)d:%(col)d: %(code)s %(text)s'],
            capture_output=True, text=True
        )
        lint_output = flake8_result.stdout.strip()
        
        # Batasi output flake8 agar tidak menghabiskan token jika errornya ribuan
        lint_lines = lint_output.split('\n')
        if len(lint_lines) > 20:
            lint_output = '\n'.join(lint_lines[:20]) + f"\n...dan {len(lint_lines)-20} masalah linting lainnya."

        # 2. RADON (Cyclomatic Complexity)
        radon_result = subprocess.run(
            ['radon', 'cc', '-s', temp_file_path],
            capture_output=True, text=True
        )
        complexity_output = radon_result.stdout.strip()

        return {
            "lint_report": lint_output if lint_output else "Flake8: Tidak ada error sintaksis/style yang signifikan (Perfect).",
            "complexity_report": complexity_output if complexity_output else "Radon: Kompleksitas sangat baik (A)."
        }
    except Exception as e:
        return {
            "lint_report": f"Gagal menjalankan linter: {e}",
            "complexity_report": f"Gagal menjalankan radon: {e}"
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def chunk_python_code(code: str, max_tokens: int = 5000) -> list[str]:
    """
    Memecah kode menjadi blok-blok fungsi/class berdasarkan AST 
    jika ukurannya melebihi batas token (Phase 1).
    """
    if estimate_tokens(code) <= max_tokens:
        return [code]
        
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Jika kode error parah, potong paksa ke tengah
        mid = len(code) // 2
        return [code[:mid], code[mid:]]

    chunks = []
    current_chunk = []
    current_tokens = 0
    lines = code.split('\n')
    
    for node in tree.body:
        start_lineno = node.lineno - 1
        end_lineno = getattr(node, 'end_lineno', start_lineno + 1)
        
        node_code = '\n'.join(lines[start_lineno:end_lineno])
        node_tokens = estimate_tokens(node_code)
        
        if current_tokens + node_tokens > max_tokens and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [node_code]
            current_tokens = node_tokens
        else:
            current_chunk.append(node_code)
            current_tokens += node_tokens
            
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        
        
    return chunks if chunks else [code]

def process_ipynb_content(ipynb_json_str: str) -> str:
    """
    Mem-parsing konten JSON dari file .ipynb.
    Menggabungkan semua cell 'code' dan membuang semua 'outputs' dan 'metadata'
    sehingga ukurannya sangat ringan. Sel 'markdown' bisa diubah jadi komentar.
    """
    try:
        notebook = json.loads(ipynb_json_str)
    except json.JSONDecodeError:
        return ""

    extracted_code = []
    cells = notebook.get('cells', [])
    
    for i, cell in enumerate(cells, 1):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        # Source bisa list of strings atau string tunggal
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source
            
        if not source_text.strip():
            continue
            
        if cell_type == 'code':
            extracted_code.append(f"# --- CELL {i} (CODE) ---")
            extracted_code.append(source_text)
            extracted_code.append("\n")
        elif cell_type == 'markdown':
            # Opsional: Kita jadikan komentar agar LLM tahu konteks narasi murid
            extracted_code.append(f"# --- CELL {i} (MARKDOWN) ---")
            for line in source_text.split('\n'):
                extracted_code.append(f"# {line}")
            extracted_code.append("\n")
            
    return "\n".join(extracted_code)
