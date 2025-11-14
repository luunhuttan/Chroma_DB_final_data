# -*- coding: utf-8 -*-
"""
Script đánh giá độ chính xác của hệ thống tìm kiếm
- Đọc queries từ random_queries.csv
- Chạy search_top5 cho mỗi query
- Người dùng nhập số câu trả lời đúng
- Tính accuracy = số đúng / 5
- Lưu progress để có thể tiếp tục sau
"""
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent
QUERIES_FILE = BASE_DIR / "random_queries.csv"
PROGRESS_FILE = BASE_DIR / "progress_final_data.json"
RESULTS_FILE = BASE_DIR / "final_results.json"
SEARCH_RESULTS_FILE = BASE_DIR / "search_results_data.json"  # File lưu kết quả tìm kiếm để đánh giá
COLLECTION_NAME = "qa_collection"
BATCH_SIZE = 20  # Số lượng queries xử lý mỗi lần chạy

# Khởi tạo ChromaDB client và collection
client = chromadb.PersistentClient(path=str(BASE_DIR / "chromadb_store"))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Khởi tạo model embedding
model = SentenceTransformer('all-MiniLM-L6-v2')


def search_top5(query: str) -> List[Dict[str, Any]]:
    """Trả về tối đa 5 hồ sơ phù hợp nhất với truy vấn."""
    q_emb = model.encode([query], convert_to_tensor=False)[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["metadatas", "distances"],
    )
    items: List[Dict[str, Any]] = []
    metas_list = (results.get("metadatas") or [[]])[0]
    distance_list = (results.get("distances") or [[]])[0]
    for idx, meta in enumerate(metas_list):
        distance = distance_list[idx] if idx < len(distance_list) else None
        items.append({
            "title": meta.get("title", ""),
            "skills": meta.get("skills", ""),
            "abilities": meta.get("abilities", ""),
            "program": meta.get("program", ""),
            "distance": distance,
        })
    return items


def load_queries() -> List[Dict[str, str]]:
    """Đọc tất cả queries từ file CSV."""
    queries = []
    if not QUERIES_FILE.exists():
        raise FileNotFoundError(f"File not found: {QUERIES_FILE}")
    
    with QUERIES_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query_id": row.get("query_id", ""),
                "query_text": row.get("query_text", ""),
                "category": row.get("category", ""),
                "target_person_id": row.get("target_person_id", ""),
                "difficulty": row.get("difficulty", ""),
            })
    return queries


def load_progress() -> Dict[str, Any]:
    """Tải progress đã lưu (nếu có)."""
    if PROGRESS_FILE.exists():
        with PROGRESS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_processed_index": 0,
        "results": []
    }


def save_progress(progress: Dict[str, Any]):
    """Lưu progress vào file."""
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_results(results: List[Dict[str, Any]]):
    """Lưu kết quả cuối cùng vào file."""
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_search_results() -> List[Dict[str, Any]]:
    """Tải kết quả tìm kiếm đã lưu (nếu có)."""
    if SEARCH_RESULTS_FILE.exists():
        with SEARCH_RESULTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_search_results(search_results_data: List[Dict[str, Any]]):
    """Lưu kết quả tìm kiếm vào file để đánh giá."""
    with SEARCH_RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(search_results_data, f, ensure_ascii=False, indent=2)


def display_results(results: List[Dict[str, Any]], query_info: Dict[str, str]):
    """Hiển thị kết quả tìm kiếm cho người dùng."""
    print("\n" + "="*80)
    print(f"Query ID: {query_info['query_id']}")
    print(f"Query: {query_info['query_text']}")
    print(f"Category: {query_info['category']} | Difficulty: {query_info['difficulty']}")
    print(f"Target Person ID: {query_info['target_person_id']}")
    print("-"*80)
    print("Top 5 kết quả:")
    for idx, result in enumerate(results, 1):
        print(f"\n[{idx}] Distance: {result.get('distance', 'N/A'):.4f}")
        print(f"    Title: {result.get('title', 'N/A')}")
        print(f"    Skills: {result.get('skills', 'N/A')[:100]}...")
        print(f"    Abilities: {result.get('abilities', 'N/A')[:100]}...")
        print(f"    Program: {result.get('program', 'N/A')}")
    print("="*80)


def get_correct_count() -> int:
    """Yêu cầu người dùng nhập số câu trả lời đúng."""
    while True:
        try:
            count = input("\nNhập số câu trả lời ĐÚNG (0-5): ").strip()
            if count.lower() in ['exit', 'quit', 'q']:
                return -1  # Signal để dừng
            count = int(count)
            if 0 <= count <= 5:
                return count
            else:
                print("Vui lòng nhập số từ 0 đến 5!")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ!")


def process_queries():
    """Xử lý queries từ file CSV."""
    # Đọc queries
    print("Đang đọc queries từ file...")
    all_queries = load_queries()
    print(f"Tổng số queries: {len(all_queries)}")
    
    # Tải progress
    progress = load_progress()
    start_index = progress["last_processed_index"]
    results = progress["results"]
    
    # Tải kết quả tìm kiếm đã lưu
    search_results_data = load_search_results()
    
    print(f"\nTiếp tục từ query thứ {start_index + 1} (đã xử lý {len(results)} queries)")
    
    # Xác định số queries cần xử lý trong batch này
    end_index = min(start_index + BATCH_SIZE, len(all_queries))
    queries_to_process = all_queries[start_index:end_index]
    
    print(f"Sẽ xử lý {len(queries_to_process)} queries (từ {start_index + 1} đến {end_index})")
    print(f"Bạn có thể nhập 'exit' hoặc 'quit' bất cứ lúc nào để dừng và lưu progress\n")
    
    # Xử lý từng query
    for idx, query_info in enumerate(queries_to_process, start=start_index):
        query_text = query_info["query_text"]
        
        if not query_text.strip():
            print(f"\nQuery {idx + 1} bỏ qua (query_text rỗng)")
            continue
        
        print(f"\n[{idx + 1}/{len(all_queries)}] Đang xử lý query...")
        
        # Tìm kiếm
        search_results = search_top5(query_text)
        
        # Lưu kết quả tìm kiếm vào file để đánh giá
        search_result_entry = {
            "query_id": query_info["query_id"],
            "query_text": query_text,
            "category": query_info["category"],
            "target_person_id": query_info["target_person_id"],
            "difficulty": query_info["difficulty"],
            "search_results": search_results,
            "timestamp": None  # Có thể thêm timestamp nếu cần
        }
        # Kiểm tra xem query_id đã tồn tại chưa (tránh trùng lặp khi tiếp tục)
        existing_idx = next((i for i, item in enumerate(search_results_data) 
                            if item.get("query_id") == query_info["query_id"]), None)
        if existing_idx is not None:
            search_results_data[existing_idx] = search_result_entry
        else:
            search_results_data.append(search_result_entry)
        save_search_results(search_results_data)
        
        if not search_results:
            print("Không tìm thấy kết quả nào!")
            correct_count = 0
            accuracy = 0.0
        else:
            # Hiển thị kết quả
            display_results(search_results, query_info)
            
            # Yêu cầu nhập số câu đúng
            correct_count = get_correct_count()
            
            if correct_count == -1:
                print("\nĐã dừng. Đang lưu progress...")
                progress["last_processed_index"] = idx
                progress["results"] = results
                save_progress(progress)
                print(f"Đã lưu progress. Đã xử lý {len(results)} queries.")
                print(f"Đã lưu kết quả tìm kiếm vào: {SEARCH_RESULTS_FILE}")
                return
            
            # Tính accuracy
            accuracy = correct_count / 5.0
        
        # Lưu kết quả
        result_entry = {
            "query_id": query_info["query_id"],
            "query_text": query_text,
            "category": query_info["category"],
            "target_person_id": query_info["target_person_id"],
            "difficulty": query_info["difficulty"],
            "correct_count": correct_count,
            "accuracy": accuracy,
            "search_results": search_results
        }
        results.append(result_entry)
        
        # Lưu progress sau mỗi query
        progress["last_processed_index"] = idx + 1
        progress["results"] = results
        save_progress(progress)
        
        print(f"✓ Đã lưu. Accuracy: {accuracy:.2%} ({correct_count}/5)")
        print(f"✓ Đã lưu kết quả tìm kiếm vào file: {SEARCH_RESULTS_FILE.name}")
    
    # Kiểm tra xem đã xử lý hết chưa
    if end_index >= len(all_queries):
        print("\n" + "="*80)
        print("ĐÃ XỬ LÝ HẾT TẤT CẢ QUERIES!")
        print("="*80)
        
        # Tính thống kê
        total = len(results)
        if total > 0:
            avg_accuracy = sum(r["accuracy"] for r in results) / total
            print(f"\nTổng số queries đã xử lý: {total}")
            print(f"Độ chính xác trung bình: {avg_accuracy:.2%}")
            
            # Thống kê theo category
            category_stats = {}
            for r in results:
                cat = r["category"]
                if cat not in category_stats:
                    category_stats[cat] = {"count": 0, "total_accuracy": 0}
                category_stats[cat]["count"] += 1
                category_stats[cat]["total_accuracy"] += r["accuracy"]
            
            print("\nThống kê theo category:")
            for cat, stats in category_stats.items():
                avg = stats["total_accuracy"] / stats["count"]
                print(f"  {cat}: {avg:.2%} (n={stats['count']})")
            
            # Thống kê theo difficulty
            difficulty_stats = {}
            for r in results:
                diff = r["difficulty"]
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {"count": 0, "total_accuracy": 0}
                difficulty_stats[diff]["count"] += 1
                difficulty_stats[diff]["total_accuracy"] += r["accuracy"]
            
            print("\nThống kê theo difficulty:")
            for diff, stats in difficulty_stats.items():
                avg = stats["total_accuracy"] / stats["count"]
                print(f"  {diff}: {avg:.2%} (n={stats['count']})")
        
        # Lưu kết quả cuối cùng
        save_results(results)
        print(f"\nĐã lưu kết quả vào: {RESULTS_FILE}")
        print(f"Đã lưu kết quả tìm kiếm để đánh giá vào: {SEARCH_RESULTS_FILE}")
        
        # Xóa file progress vì đã xong
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            print("Đã xóa file progress.")
    else:
        print(f"\nĐã xử lý {len(queries_to_process)} queries trong batch này.")
        print(f"Còn lại {len(all_queries) - end_index} queries.")
        print(f"Chạy lại script để tiếp tục từ query {end_index + 1}")


if __name__ == "__main__":
    try:
        process_queries()
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng (Ctrl+C). Progress đã được lưu.")
    except Exception as e:
        print(f"\nLỗi: {e}")
        import traceback
        traceback.print_exc()

