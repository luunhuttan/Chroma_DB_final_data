# -*- coding: utf-8 -*-
"""
Script đánh giá độ chính xác của hệ thống tìm kiếm
- Đọc queries từ random_queries.csv
- Chạy search_top5 cho mỗi query
- Tính Precision@K, AP@K, MAP@K dựa trên distance hoặc relevance score
- Lưu progress để có thể tiếp tục sau
"""
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent
QUERIES_FILE = BASE_DIR / "random_queries.csv"
PROGRESS_FILE = BASE_DIR / "progress_final_data.json"
RESULTS_FILE = BASE_DIR / "final_results.json"
SEARCH_RESULTS_FILE = BASE_DIR / "search_results_data.json"  # File lưu kết quả tìm kiếm để đánh giá
COLLECTION_NAME = "qa_collection"
BATCH_SIZE = 20  # Số lượng queries xử lý mỗi lần chạy
AUTO_EVALUATION = True  # True = tự động đánh giá, False = đánh giá thủ công
EVALUATION_METHOD = "distance"  # "distance" hoặc "relevance"
DISTANCE_THRESHOLD = 0.8  # Nếu dùng distance: distance < 0.8 → relevant
RELEVANCE_THRESHOLD = 0.5  # Nếu dùng relevance: score >= 0.5 → relevant

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
    # ids luôn được trả về trong response, không cần include
    ids_list = (results.get("ids") or [[]])[0]
    for idx, meta in enumerate(metas_list):
        distance = distance_list[idx] if idx < len(distance_list) else None
        person_id = ids_list[idx] if idx < len(ids_list) else None
        items.append({
            "person_id": person_id,
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


def display_results(results: List[Dict[str, Any]], query_info: Dict[str, str], query_text: str = ""):
    """Hiển thị kết quả tìm kiếm cho người dùng với đánh giá đúng/khớp."""
    print("\n" + "="*80)
    print(f"QUERY THÔNG TIN")
    print("="*80)
    print(f"Query ID: {query_info['query_id']}")
    print(f"Query: {query_info['query_text']}")
    print(f"Category: {query_info['category']} | Difficulty: {query_info['difficulty']}")
    print("\n" + "-"*80)
    print("TOP 5 KẾT QUẢ TÌM KIẾM")
    print("-"*80)
    
    # Tính relevance labels để hiển thị
    if EVALUATION_METHOD == "distance":
        threshold = DISTANCE_THRESHOLD
        method_desc = f"distance < {threshold}"
    else:
        threshold = RELEVANCE_THRESHOLD
        method_desc = f"relevance >= {threshold}"
    
    print(f"\n📊 Tiêu chí đánh giá: {method_desc}")
    if EVALUATION_METHOD == "distance":
        print("   💡 Distance càng NHỎ → càng giống → càng ĐÚNG")
    else:
        print("   💡 Relevance score càng CAO → càng phù hợp → càng ĐÚNG")
    print("\n" + "-"*80)
    
    for idx, result in enumerate(results, 1):
        person_id = result.get('person_id', 'N/A')
        distance = result.get('distance', None)
        title = result.get('title', 'N/A')
        skills = result.get('skills', 'N/A')
        abilities = result.get('abilities', 'N/A')
        program = result.get('program', 'N/A')
        
        # Xác định xem kết quả có đúng/khớp không
        is_relevant = False
        if EVALUATION_METHOD == "distance":
            if distance is not None:
                is_relevant = distance < threshold
        else:
            if query_text:
                score = calculate_relevance_score(query_text, result)
                is_relevant = score >= threshold
        
        # Đánh dấu rõ ràng
        status_icon = "✓ ĐÚNG/KHỚP" if is_relevant else "✗ KHÔNG KHỚP"
        status_color = "✓" if is_relevant else "✗"
        
        print(f"\n{'='*80}")
        print(f"KẾT QUẢ [{idx}/5] - {status_icon}")
        print(f"{'='*80}")
        print(f"Person ID: {person_id}")
        if distance is not None:
            relevance_info = f"Distance: {distance:.4f}"
            if EVALUATION_METHOD == "distance":
                relevance_info += f" {'✓' if is_relevant else '✗'} {'< ' if is_relevant else '≥ '}{threshold}"
            else:
                if query_text:
                    score = calculate_relevance_score(query_text, result)
                    relevance_info += f" | Relevance: {score:.3f} {'✓' if is_relevant else '✗'} {'≥ ' if is_relevant else '< '}{threshold}"
            print(f"{relevance_info}")
        print(f"\n📋 Title/Role:")
        print(f"   {title}")
        print(f"\n🛠️  Skills:")
        # Hiển thị đầy đủ skills, chia thành nhiều dòng nếu quá dài
        if skills and skills != 'N/A':
            # Chia thành các dòng 80 ký tự
            words = skills.split(', ')
            line = ""
            for word in words:
                if len(line) + len(word) + 2 > 75:
                    if line:
                        print(f"   {line.strip()}")
                    line = word + ", "
                else:
                    line += word + ", "
            if line:
                print(f"   {line.rstrip(', ')}")
        else:
            print(f"   {skills}")
        print(f"\n💼 Abilities:")
        if abilities and abilities != 'N/A':
            # Chia thành các dòng 80 ký tự
            words = abilities.split(', ')
            line = ""
            for word in words:
                if len(line) + len(word) + 2 > 75:
                    if line:
                        print(f"   {line.strip()}")
                    line = word + ", "
                else:
                    line += word + ", "
            if line:
                print(f"   {line.rstrip(', ')}")
        else:
            print(f"   {abilities}")
        print(f"\n🎓 Education/Program:")
        print(f"   {program}")
    
    print("\n" + "="*80)


def extract_keywords(text: str) -> set:
    """Trích xuất keywords từ text (loại bỏ stop words đơn giản)."""
    # Chuyển thành lowercase và tách thành từ
    words = re.findall(r'\b\w+\b', text.lower())
    # Loại bỏ các từ quá ngắn (< 3 ký tự) và các từ thường gặp
    stop_words = {'the', 'for', 'and', 'with', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'of', 'from', 'by', 'as', 'or', 'but', 'not', 'this', 'that', 'these', 'those'}
    keywords = {w for w in words if len(w) >= 3 and w not in stop_words}
    return keywords


def calculate_relevance_score(query: str, result: Dict[str, Any]) -> float:
    """
    Tính điểm relevance (0-1) dựa trên:
    1. Distance (semantic similarity) - 40%
    2. Keyword matching trong title - 20%
    3. Keyword matching trong skills - 25%
    4. Keyword matching trong abilities - 15%
    """
    score = 0.0
    
    # 1. Distance score (càng nhỏ càng tốt, normalize về 0-1)
    distance = result.get('distance')
    if distance is not None:
        # Distance thường trong khoảng 0-2, normalize
        # Distance nhỏ = similarity cao
        distance_score = max(0, 1 - (distance / 2.0))  # Nếu distance = 0 -> score = 1, distance = 2 -> score = 0
        score += distance_score * 0.4
    
    # 2-4. Keyword matching
    query_keywords = extract_keywords(query)
    
    # Title matching (20%)
    title = result.get('title', '').lower()
    title_keywords = extract_keywords(title)
    title_match = len(query_keywords & title_keywords) / max(len(query_keywords), 1)
    score += title_match * 0.2
    
    # Skills matching (25%)
    skills = result.get('skills', '').lower()
    skills_keywords = extract_keywords(skills)
    skills_match = len(query_keywords & skills_keywords) / max(len(query_keywords), 1)
    score += skills_match * 0.25
    
    # Abilities matching (15%)
    abilities = result.get('abilities', '').lower()
    abilities_keywords = extract_keywords(abilities)
    abilities_match = len(query_keywords & abilities_keywords) / max(len(query_keywords), 1)
    score += abilities_match * 0.15
    
    return min(1.0, score)


def get_relevance_labels(results: List[Dict[str, Any]], query: str, 
                         method: str = "distance", threshold: float = 0.8) -> List[int]:
    """
    Xác định relevance label (0 hoặc 1) cho mỗi kết quả.
    
    Args:
        results: List of search results
        query: Query text (để tính relevance score nếu method="relevance")
        method: "distance" (chỉ dùng distance) hoặc "relevance" (dùng relevance score)
        threshold: 
            - Nếu method="distance": distance threshold (càng nhỏ càng tốt, thường 0.6-1.0)
            - Nếu method="relevance": relevance score threshold (0-1, càng cao càng tốt)
    
    Returns:
        List of binary relevance labels [0, 1, 0, 1, ...]
    """
    labels = []
    
    for result in results:
        is_relevant = False
        
        if method == "distance":
            # Chỉ dùng distance: distance càng nhỏ = càng giống = càng phù hợp
            distance = result.get('distance')
            if distance is not None:
                # Distance < threshold → relevant
                is_relevant = distance < threshold
        elif method == "relevance":
            # Dùng relevance score (kết hợp distance + keyword matching)
            score = calculate_relevance_score(query, result)
            is_relevant = score >= threshold
        
        labels.append(1 if is_relevant else 0)
    
    return labels


def precision_at_k(relevance_labels: List[int], k: int) -> float:
    """
    Tính Precision@K.
    
    Args:
        relevance_labels: List of binary relevance labels [0, 1, 0, 1, ...]
        k: Số kết quả đầu tiên cần xem xét
    
    Returns:
        Precision@K (0.0 - 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k_labels = relevance_labels[:k]
    if not top_k_labels:
        return 0.0
    
    relevant_count = sum(top_k_labels)
    return relevant_count / len(top_k_labels)


def average_precision_at_k(relevance_labels: List[int], k: int) -> float:
    """
    Tính Average Precision@K (AP@K).
    
    AP@K = (1/R) * sum(P@i for i where result i is relevant)
    R = tổng số kết quả relevant trong top K
    
    Args:
        relevance_labels: List of binary relevance labels
        k: Số kết quả đầu tiên cần xem xét
    
    Returns:
        AP@K (0.0 - 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k_labels = relevance_labels[:k]
    if not top_k_labels:
        return 0.0
    
    # Tổng số kết quả relevant trong top K
    total_relevant = sum(top_k_labels)
    if total_relevant == 0:
        return 0.0
    
    # Tính precision tại mỗi vị trí có kết quả relevant
    ap_sum = 0.0
    relevant_found = 0
    
    for i, label in enumerate(top_k_labels, 1):
        if label == 1:  # Kết quả này là relevant
            relevant_found += 1
            # Precision tại vị trí i = số relevant từ đầu đến i / i
            precision_at_i = relevant_found / i
            ap_sum += precision_at_i
    
    return ap_sum / total_relevant


def calculate_metrics(results: List[Dict[str, Any]], query: str,
                     k: int = 5, method: str = "distance", threshold: float = 0.8) -> Dict[str, float]:
    """
    Tính các metrics: Precision@K, AP@K.
    Đánh giá dựa trên distance hoặc relevance score.
    
    Args:
        results: List of search results
        query: Query text (để tính relevance score nếu method="relevance")
        k: Số kết quả đầu tiên (mặc định 5)
        method: "distance" (chỉ dùng distance) hoặc "relevance" (dùng relevance score)
        threshold: 
            - Nếu method="distance": distance threshold (thường 0.6-1.0)
            - Nếu method="relevance": relevance score threshold (0-1)
    
    Returns:
        Dict chứa các metrics
    """
    relevance_labels = get_relevance_labels(results, query, method, threshold)
    
    p_at_k = precision_at_k(relevance_labels, k)
    ap_at_k = average_precision_at_k(relevance_labels, k)
    
    return {
        'precision_at_k': p_at_k,
        'ap_at_k': ap_at_k,
        'relevance_labels': relevance_labels,
        'num_relevant': sum(relevance_labels)
    }


def auto_evaluate_results(query: str, results: List[Dict[str, Any]], method: str = "combined", threshold: float = 0.5) -> int:
    """
    Tự động đánh giá số kết quả phù hợp.
    
    Args:
        query: Query text
        results: List of search results
        method: "distance", "keywords", hoặc "combined"
        threshold: Ngưỡng để coi là phù hợp (0-1)
    
    Returns:
        Số kết quả được đánh giá là phù hợp (0-5)
    """
    correct_count = 0
    
    for result in results:
        is_relevant = False
        
        if method == "distance":
            # Chỉ dựa trên distance
            distance = result.get('distance')
            if distance is not None:
                # Distance < threshold * 2 (vì distance thường 0-2)
                is_relevant = distance < (threshold * 2)
        
        elif method == "keywords":
            # Chỉ dựa trên keyword matching
            score = calculate_relevance_score(query, result)
            is_relevant = score >= threshold
        
        elif method == "combined":
            # Kết hợp distance và keywords
            score = calculate_relevance_score(query, result)
            is_relevant = score >= threshold
        
        if is_relevant:
            correct_count += 1
    
    return correct_count


def get_correct_count(query: str, results: List[Dict[str, Any]], auto_mode: bool = False) -> int:
    """
    Đánh giá số câu trả lời đúng.
    Nếu auto_mode=True, tự động đánh giá. Nếu False, yêu cầu người dùng nhập.
    """
    if auto_mode:
        # Tự động đánh giá
        print("\n" + "="*80)
        print("ĐÁNH GIÁ TỰ ĐỘNG")
        print("="*80)
        
        if EVALUATION_METHOD == "distance":
            print(f"Đang đánh giá tự động dựa trên DISTANCE (distance < {DISTANCE_THRESHOLD} → relevant)")
            print("  💡 Distance càng NHỎ → càng giống → càng đúng")
            print("\nDistance của từng kết quả:")
            for idx, result in enumerate(results, 1):
                distance = result.get('distance', 'N/A')
                person_id = result.get('person_id', 'N/A')
                is_relevant = distance != 'N/A' and distance < DISTANCE_THRESHOLD
                status = "✓ RELEVANT" if is_relevant else "✗ Non-relevant"
                print(f"  [{idx}] Person ID: {person_id} | Distance: {distance:.4f} {status}")
            
            # Đếm số relevant
            correct_count = sum(1 for r in results 
                              if r.get('distance') is not None and r.get('distance') < DISTANCE_THRESHOLD)
            print(f"\n✓ Tự động đánh giá: {correct_count}/5 kết quả phù hợp (distance < {DISTANCE_THRESHOLD})")
        else:
            print(f"Đang đánh giá tự động dựa trên RELEVANCE SCORE (score >= {RELEVANCE_THRESHOLD} → relevant)")
            print("  💡 Relevance score càng CAO → càng phù hợp → càng đúng")
            print("\nRelevance score của từng kết quả:")
            for idx, result in enumerate(results, 1):
                score = calculate_relevance_score(query, result)
                distance = result.get('distance', 'N/A')
                person_id = result.get('person_id', 'N/A')
                is_relevant = score >= RELEVANCE_THRESHOLD
                status = "✓ RELEVANT" if is_relevant else "✗ Non-relevant"
                print(f"  [{idx}] Person ID: {person_id} | Distance: {distance:.4f} | Relevance: {score:.3f} {status}")
            
            # Đếm số relevant
            correct_count = sum(1 for r in results 
                              if calculate_relevance_score(query, r) >= RELEVANCE_THRESHOLD)
            print(f"\n✓ Tự động đánh giá: {correct_count}/5 kết quả phù hợp (relevance >= {RELEVANCE_THRESHOLD})")
        
        # Cho phép người dùng xác nhận hoặc chỉnh sửa
        print("\nBạn có muốn chỉnh sửa kết quả này không? (Enter để chấp nhận, hoặc nhập số 0-5):")
        user_input = input(">>> ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            return -1
        elif user_input == "":
            return correct_count
        else:
            try:
                count = int(user_input)
                if 0 <= count <= 5:
                    return count
                else:
                    print("⚠️  Số không hợp lệ, sử dụng kết quả tự động.")
                    return correct_count
            except ValueError:
                print("⚠️  Không hợp lệ, sử dụng kết quả tự động.")
                return correct_count
    else:
        # Đánh giá thủ công
        print("\n" + "="*80)
        print("ĐÁNH GIÁ KẾT QUẢ")
        print("="*80)
        print("Dựa trên các tiêu chí đã gợi ý ở trên, hãy đếm số kết quả PHÙ HỢP với query.")
        print("Một kết quả được coi là PHÙ HỢP nếu:")
        print("  ✓ Có các kỹ năng/công nghệ được yêu cầu trong query")
        print("  ✓ Chức danh/vai trò phù hợp với yêu cầu")
        print("  ✓ Bằng cấp/giáo dục phù hợp (nếu query yêu cầu)")
        print("  ✓ Có độ liên quan tổng thể tốt với query")
        print("\nNhập số kết quả PHÙ HỢP (0-5):")
        while True:
            try:
                count = input(">>> ").strip()
                if count.lower() in ['exit', 'quit', 'q']:
                    return -1  # Signal để dừng
                count = int(count)
                if 0 <= count <= 5:
                    return count
                else:
                    print("⚠️  Vui lòng nhập số từ 0 đến 5!")
            except ValueError:
                print("⚠️  Vui lòng nhập một số hợp lệ!")


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
            metrics = {
                'precision_at_k': 0.0,
                'ap_at_k': 0.0,
                'relevance_labels': [0, 0, 0, 0, 0],
                'num_relevant': 0
            }
        else:
            # Hiển thị kết quả (truyền query_text để tính relevance nếu cần)
            display_results(search_results, query_info, query_text)
            
            # Tính metrics (Precision@K, AP@K)
            if EVALUATION_METHOD == "distance":
                threshold = DISTANCE_THRESHOLD
                method_desc = f"distance < {threshold}"
            else:
                threshold = RELEVANCE_THRESHOLD
                method_desc = f"relevance score >= {threshold}"
            
            metrics = calculate_metrics(
                search_results, 
                query=query_text,
                k=5,
                method=EVALUATION_METHOD,
                threshold=threshold
            )
            
            # Hiển thị metrics
            print("\n" + "="*80)
            print(f"METRICS ĐÁNH GIÁ (dựa trên {EVALUATION_METHOD}, {method_desc})")
            print("="*80)
            print(f"Precision@5: {metrics['precision_at_k']:.4f} ({metrics['num_relevant']}/5 relevant)")
            print(f"AP@5 (Average Precision@5): {metrics['ap_at_k']:.4f}")
            print(f"Relevance labels: {metrics['relevance_labels']}")
            
            print(f"\n💡 CÁC METRICS ĐƯỢC TÍNH DỰA TRÊN:")
            print(f"   1. Precision@K: Tỷ lệ kết quả relevant trong top K")
            print(f"   2. AP@K (Average Precision@K): Trung bình precision tại các vị trí có kết quả relevant")
            print(f"   3. MAP@K (Mean Average Precision@K): Trung bình của tất cả AP@K qua tất cả queries")
            print(f"   (MAP@K sẽ được hiển thị khi hoàn thành tất cả queries)")
            
            if EVALUATION_METHOD == "distance":
                print(f"\n📊 Tiêu chí xác định relevant: distance < {DISTANCE_THRESHOLD}")
                print(f"   (Distance càng nhỏ → càng giống → càng đúng)")
            else:
                print(f"\n📊 Tiêu chí xác định relevant: relevance score >= {RELEVANCE_THRESHOLD}")
                print(f"   (Relevance score = distance 40% + keyword matching 60%)")
            
            # Đánh giá (tự động hoặc thủ công) - giữ lại để tương thích (không dùng cho metrics)
            correct_count = get_correct_count(query_text, search_results, auto_mode=AUTO_EVALUATION)
            
            if correct_count == -1:
                print("\nĐã dừng. Đang lưu progress...")
                progress["last_processed_index"] = idx
                progress["results"] = results
                save_progress(progress)
                print(f"Đã lưu progress. Đã xử lý {len(results)} queries.")
                print(f"Đã lưu kết quả tìm kiếm vào: {SEARCH_RESULTS_FILE}")
                return
        
        # Lưu kết quả (chỉ dùng metrics, không dùng accuracy)
        result_entry = {
            "query_id": query_info["query_id"],
            "query_text": query_text,
            "category": query_info["category"],
            "target_person_id": query_info["target_person_id"],
            "difficulty": query_info["difficulty"],
            "precision_at_5": metrics['precision_at_k'],
            "ap_at_5": metrics['ap_at_k'],
            "relevance_labels": metrics['relevance_labels'],
            "num_relevant": metrics['num_relevant'],
            "search_results": search_results
        }
        results.append(result_entry)
        
        # Lưu progress sau mỗi query
        progress["last_processed_index"] = idx + 1
        progress["results"] = results
        save_progress(progress)
        
        print(f"✓ Đã lưu. Precision@5: {metrics['precision_at_k']:.4f} ({metrics['num_relevant']}/5)")
        print(f"✓ Đã lưu kết quả tìm kiếm vào file: {SEARCH_RESULTS_FILE.name}")
    
    # Kiểm tra xem đã xử lý hết chưa
    if end_index >= len(all_queries):
        print("\n" + "="*80)
        print("ĐÃ XỬ LÝ HẾT TẤT CẢ QUERIES!")
        print("="*80)
        
        # Tính thống kê
        total = len(results)
        if total > 0:
            # Tính MAP@5 (Mean Average Precision@5)
            ap_scores = [r.get("ap_at_5", 0.0) for r in results]
            map_at_5 = sum(ap_scores) / total if total > 0 else 0.0
            
            # Tính Precision@5 trung bình
            precision_scores = [r.get("precision_at_5", 0.0) for r in results]
            avg_precision_at_5 = sum(precision_scores) / total if total > 0 else 0.0
            
            # Phân tích phân phối Precision@5
            precision_sorted = sorted(precision_scores)
            min_precision = min(precision_scores)
            max_precision = max(precision_scores)
            median_precision = precision_sorted[total // 2] if total > 0 else 0.0
            
            # Đếm số queries theo mức Precision@5
            perfect_queries = sum(1 for p in precision_scores if p == 1.0)
            high_queries = sum(1 for p in precision_scores if 0.8 <= p < 1.0)
            medium_queries = sum(1 for p in precision_scores if 0.5 <= p < 0.8)
            low_queries = sum(1 for p in precision_scores if p < 0.5)
            
            # Phân tích phân phối AP@5
            ap_sorted = sorted(ap_scores)
            min_ap = min(ap_scores)
            max_ap = max(ap_scores)
            median_ap = ap_sorted[total // 2] if total > 0 else 0.0
            
            # Phân tích số lượng relevant results
            all_num_relevant = [r.get("num_relevant", 0) for r in results]
            avg_relevant = sum(all_num_relevant) / total if total > 0 else 0.0
            total_relevant_all = sum(all_num_relevant)
            max_possible = total * 5  # Mỗi query có 5 kết quả
            
            # Phân tích distance (nếu có)
            all_distances = []
            for r in results:
                for sr in r.get("search_results", []):
                    dist = sr.get("distance")
                    if dist is not None:
                        all_distances.append(dist)
            
            print(f"\nTổng số queries đã xử lý: {total}")
            print(f"\n{'='*80}")
            print("METRICS TỔNG KẾT")
            print(f"{'='*80}")
            print(f"Precision@5 trung bình: {avg_precision_at_5:.4f}")
            print(f"MAP@5 (Mean Average Precision@5): {map_at_5:.4f}")
            
            print(f"\n{'='*80}")
            print("PHÂN TÍCH CHI TIẾT PRECISION@5")
            print(f"{'='*80}")
            print(f"Min: {min_precision:.4f} | Max: {max_precision:.4f} | Median: {median_precision:.4f}")
            print(f"\nPhân bố Precision@5:")
            print(f"  Perfect (1.0000): {perfect_queries} queries ({perfect_queries/total*100:.1f}%)")
            print(f"  High (0.80-0.99): {high_queries} queries ({high_queries/total*100:.1f}%)")
            print(f"  Medium (0.50-0.79): {medium_queries} queries ({medium_queries/total*100:.1f}%)")
            print(f"  Low (<0.50): {low_queries} queries ({low_queries/total*100:.1f}%)")
            
            print(f"\n{'='*80}")
            print("PHÂN TÍCH CHI TIẾT AP@5")
            print(f"{'='*80}")
            print(f"Min: {min_ap:.4f} | Max: {max_ap:.4f} | Median: {median_ap:.4f}")
            
            print(f"\n{'='*80}")
            print("PHÂN TÍCH SỐ LƯỢNG RELEVANT RESULTS")
            print(f"{'='*80}")
            print(f"Tổng số kết quả relevant: {total_relevant_all}/{max_possible}")
            print(f"Tỷ lệ relevant: {total_relevant_all/max_possible*100:.2f}%")
            print(f"Số lượng relevant trung bình mỗi query: {avg_relevant:.2f}/5")
            
            if all_distances:
                avg_distance = sum(all_distances) / len(all_distances)
                min_distance = min(all_distances)
                max_distance = max(all_distances)
                distance_sorted = sorted(all_distances)
                median_distance = distance_sorted[len(distance_sorted) // 2]
                
                print(f"\n{'='*80}")
                print("PHÂN TÍCH DISTANCE")
                print(f"{'='*80}")
                print(f"Distance trung bình: {avg_distance:.4f}")
                print(f"Min: {min_distance:.4f} | Max: {max_distance:.4f} | Median: {median_distance:.4f}")
                if EVALUATION_METHOD == "distance":
                    relevant_distances = [d for d in all_distances if d < DISTANCE_THRESHOLD]
                    non_relevant_distances = [d for d in all_distances if d >= DISTANCE_THRESHOLD]
                    print(f"\nVới threshold = {DISTANCE_THRESHOLD}:")
                    print(f"  Relevant: {len(relevant_distances)} kết quả ({len(relevant_distances)/len(all_distances)*100:.1f}%)")
                    print(f"  Non-relevant: {len(non_relevant_distances)} kết quả ({len(non_relevant_distances)/len(all_distances)*100:.1f}%)")
                    if relevant_distances:
                        print(f"  Distance trung bình của relevant: {sum(relevant_distances)/len(relevant_distances):.4f}")
                    if non_relevant_distances:
                        print(f"  Distance trung bình của non-relevant: {sum(non_relevant_distances)/len(non_relevant_distances):.4f}")
            
            # Thống kê theo category
            category_stats = {}
            for r in results:
                cat = r["category"]
                if cat not in category_stats:
                    category_stats[cat] = {
                        "count": 0, 
                        "total_precision": 0,
                        "total_ap": 0
                    }
                category_stats[cat]["count"] += 1
                category_stats[cat]["total_precision"] += r.get("precision_at_5", 0.0)
                category_stats[cat]["total_ap"] += r.get("ap_at_5", 0.0)
            
            # Top queries tốt nhất và xấu nhất
            results_with_scores = [(r, r.get("precision_at_5", 0.0), r.get("ap_at_5", 0.0)) for r in results]
            results_with_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            print(f"\n{'='*80}")
            print("TOP 5 QUERIES TỐT NHẤT (theo Precision@5)")
            print(f"{'='*80}")
            for i, (r, prec, ap) in enumerate(results_with_scores[:5], 1):
                print(f"{i}. Query ID: {r['query_id']} | Category: {r['category']} | Difficulty: {r['difficulty']}")
                print(f"   Precision@5: {prec:.4f} | AP@5: {ap:.4f} | Relevant: {r.get('num_relevant', 0)}/5")
                print(f"   Query: {r['query_text'][:80]}...")
            
            print(f"\n{'='*80}")
            print("TOP 5 QUERIES XẤU NHẤT (theo Precision@5)")
            print(f"{'='*80}")
            for i, (r, prec, ap) in enumerate(results_with_scores[-5:], 1):
                print(f"{i}. Query ID: {r['query_id']} | Category: {r['category']} | Difficulty: {r['difficulty']}")
                print(f"   Precision@5: {prec:.4f} | AP@5: {ap:.4f} | Relevant: {r.get('num_relevant', 0)}/5")
                print(f"   Query: {r['query_text'][:80]}...")
            
            print(f"\n{'='*80}")
            print("THỐNG KÊ THEO CATEGORY")
            print(f"{'='*80}")
            for cat, stats in sorted(category_stats.items()):
                avg_prec = stats["total_precision"] / stats["count"]
                avg_ap = stats["total_ap"] / stats["count"]
                # Tính min, max cho category này
                cat_precisions = [r.get("precision_at_5", 0.0) for r in results if r["category"] == cat]
                cat_min = min(cat_precisions) if cat_precisions else 0.0
                cat_max = max(cat_precisions) if cat_precisions else 0.0
                cat_perfect = sum(1 for p in cat_precisions if p == 1.0)
                print(f"  {cat} (n={stats['count']}):")
                print(f"    Precision@5: {avg_prec:.4f} (min: {cat_min:.4f}, max: {cat_max:.4f}, perfect: {cat_perfect})")
                print(f"    AP@5: {avg_ap:.4f}")
            
            # Thống kê theo difficulty
            difficulty_stats = {}
            for r in results:
                diff = r["difficulty"]
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {
                        "count": 0, 
                        "total_precision": 0,
                        "total_ap": 0
                    }
                difficulty_stats[diff]["count"] += 1
                difficulty_stats[diff]["total_precision"] += r.get("precision_at_5", 0.0)
                difficulty_stats[diff]["total_ap"] += r.get("ap_at_5", 0.0)
            
            print(f"\n{'='*80}")
            print("THỐNG KÊ THEO DIFFICULTY")
            print(f"{'='*80}")
            for diff, stats in sorted(difficulty_stats.items()):
                avg_prec = stats["total_precision"] / stats["count"]
                avg_ap = stats["total_ap"] / stats["count"]
                # Tính min, max cho difficulty này
                diff_precisions = [r.get("precision_at_5", 0.0) for r in results if r["difficulty"] == diff]
                diff_min = min(diff_precisions) if diff_precisions else 0.0
                diff_max = max(diff_precisions) if diff_precisions else 0.0
                diff_perfect = sum(1 for p in diff_precisions if p == 1.0)
                print(f"  {diff} (n={stats['count']}):")
                print(f"    Precision@5: {avg_prec:.4f} (min: {diff_min:.4f}, max: {diff_max:.4f}, perfect: {diff_perfect})")
                print(f"    AP@5: {avg_ap:.4f}")
            
            # Phân tích và đề xuất threshold
            if EVALUATION_METHOD == "distance" and all_distances:
                print(f"\n{'='*80}")
                print("PHÂN TÍCH VÀ ĐỀ XUẤT THRESHOLD")
                print(f"{'='*80}")
                print(f"Threshold hiện tại: {DISTANCE_THRESHOLD}")
                print(f"\nPhân tích với các threshold khác nhau:")
                test_thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
                for thresh in test_thresholds:
                    relevant_count = sum(1 for d in all_distances if d < thresh)
                    relevant_pct = relevant_count / len(all_distances) * 100
                    print(f"  Threshold {thresh:.2f}: {relevant_count}/{len(all_distances)} relevant ({relevant_pct:.1f}%)")
                
                # Đề xuất threshold dựa trên phân vị
                if len(all_distances) >= 10:
                    p25 = distance_sorted[len(distance_sorted) // 4]
                    p50 = median_distance
                    p75 = distance_sorted[len(distance_sorted) * 3 // 4]
                    print(f"\nPhân vị distance:")
                    print(f"  25th percentile (P25): {p25:.4f}")
                    print(f"  50th percentile (Median): {p50:.4f}")
                    print(f"  75th percentile (P75): {p75:.4f}")
                    print(f"\n💡 Đề xuất:")
                    print(f"  - Threshold chặt chẽ (P25): {p25:.4f} → ~{sum(1 for d in all_distances if d < p25)/len(all_distances)*100:.1f}% relevant")
                    print(f"  - Threshold vừa phải (P50): {p50:.4f} → ~{sum(1 for d in all_distances if d < p50)/len(all_distances)*100:.1f}% relevant")
                    print(f"  - Threshold lỏng (P75): {p75:.4f} → ~{sum(1 for d in all_distances if d < p75)/len(all_distances)*100:.1f}% relevant")
            
            # Tóm tắt cuối cùng
            print(f"\n{'='*80}")
            print("TÓM TẮT ĐÁNH GIÁ")
            print(f"{'='*80}")
            print(f"📊 Tổng quan:")
            print(f"   - Tổng số queries: {total}")
            print(f"   - Precision@5 trung bình: {avg_precision_at_5:.4f} ({avg_precision_at_5*100:.2f}%)")
            print(f"   - MAP@5: {map_at_5:.4f} ({map_at_5*100:.2f}%)")
            print(f"   - Số queries đạt perfect (1.0): {perfect_queries}/{total} ({perfect_queries/total*100:.1f}%)")
            print(f"   - Số queries có Precision@5 >= 0.8: {perfect_queries + high_queries}/{total} ({(perfect_queries + high_queries)/total*100:.1f}%)")
            print(f"\n📈 Chất lượng:")
            if avg_precision_at_5 >= 0.9:
                print(f"   ✓ Hệ thống hoạt động RẤT TỐT (Precision@5 >= 90%)")
            elif avg_precision_at_5 >= 0.8:
                print(f"   ✓ Hệ thống hoạt động TỐT (Precision@5 >= 80%)")
            elif avg_precision_at_5 >= 0.7:
                print(f"   ⚠ Hệ thống hoạt động KHÁ (Precision@5 >= 70%)")
            else:
                print(f"   ⚠ Hệ thống cần CẢI THIỆN (Precision@5 < 70%)")
            
            if EVALUATION_METHOD == "distance":
                print(f"\n⚙️  Cấu hình đánh giá:")
                print(f"   - Phương pháp: Distance-based")
                print(f"   - Threshold: {DISTANCE_THRESHOLD}")
                print(f"   - Tiêu chí: distance < {DISTANCE_THRESHOLD} → relevant")
            else:
                print(f"\n⚙️  Cấu hình đánh giá:")
                print(f"   - Phương pháp: Relevance score-based")
                print(f"   - Threshold: {RELEVANCE_THRESHOLD}")
                print(f"   - Tiêu chí: relevance score >= {RELEVANCE_THRESHOLD} → relevant")
        
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

