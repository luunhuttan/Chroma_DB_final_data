# -*- coding: utf-8 -*-
"""
Script populate dữ liệu vào ChromaDB
Đọc từ resume_CLEANED.csv và thêm vào collection qa_collection
"""
import csv
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        return iterable

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "resume_CLEANED.csv"
COLLECTION_NAME = "qa_collection"

# Khởi tạo model embedding (phải cùng model với final_data.py)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Kết nối ChromaDB
print("Ket noi ChromaDB...")
client = chromadb.PersistentClient(path=str(BASE_DIR / "chromadb_store"))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Kiểm tra số lượng hiện tại
current_count = collection.count()
print(f"So luong documents hien tai: {current_count}")

if current_count > 0:
    response = input(f"Collection da co {current_count} documents. Ban co muon xoa va them lai? (y/n): ")
    if response.lower() == 'y':
        print("Xoa collection cu...")
        client.delete_collection(name=COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print("Da xoa xong.")
    else:
        print("Giu nguyen collection. Them du lieu moi vao...")

# Đọc dữ liệu từ CSV
print(f"\nDang doc du lieu tu {DATA_FILE}...")
if not DATA_FILE.exists():
    print(f"LOI: Khong tim thay file {DATA_FILE}")
    exit(1)

rows = []
with DATA_FILE.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Da doc duoc {len(rows)} records")

# Chuẩn bị dữ liệu để thêm vào ChromaDB
print("\nDang chuan bi du lieu va tao embeddings...")
ids = []
embeddings = []
metadatas = []

batch_size = 100  # Xử lý theo batch để tránh hết RAM

for i in tqdm(range(0, len(rows), batch_size), desc="Xu ly batches"):
    batch_rows = rows[i:i+batch_size]
    
    batch_ids = []
    batch_texts = []
    batch_metadatas = []
    
    for row in batch_rows:
        person_id = row.get("person_id", "").strip()
        if not person_id:
            continue
        
        # Tạo text để embedding (kết hợp title, skills, abilities, program)
        title = row.get("title", "").strip()
        skills = row.get("skill", "").strip()
        abilities = row.get("ability", "").strip()
        program = row.get("program", "").strip()
        
        # Kết hợp các trường thành một text để tạo embedding
        combined_text = f"{title}. {skills}. {abilities}. {program}".strip()
        
        if not combined_text:
            continue
        
        batch_ids.append(str(person_id))
        batch_texts.append(combined_text)
        batch_metadatas.append({
            "person_id": person_id,
            "title": title,
            "skills": skills,
            "abilities": abilities,
            "program": program,
        })
    
    if not batch_texts:
        continue
    
    # Tạo embeddings cho batch này
    batch_embeddings = model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
    
    # Thêm vào ChromaDB
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings.tolist(),
        metadatas=batch_metadatas
    )

# Kiem tra ket qua
final_count = collection.count()
print(f"\nOK Hoan thanh! So luong documents trong collection: {final_count}")

# Test query
print("\nThu query de kiem tra...")
test_query = "software developer"
q_emb = model.encode([test_query], convert_to_tensor=False)[0].tolist()
results = collection.query(
    query_embeddings=[q_emb],
    n_results=3,
    include=["metadatas", "distances"]
)

if results.get("ids") and results["ids"][0]:
    print(f"OK Query test thanh cong! Tim thay {len(results['ids'][0])} ket qua")
    print(f"  Document dau tien: person_id={results['metadatas'][0][0].get('person_id')}")
else:
    print("X Query test khong tim thay ket qua")

