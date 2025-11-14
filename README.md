# Hệ thống Đánh giá Độ Chính Xác Tìm Kiếm với ChromaDB

## 📋 Mục đích

Project này được thiết kế để đánh giá độ chính xác của hệ thống tìm kiếm semantic sử dụng ChromaDB. Hệ thống sẽ:

- Đọc các câu truy vấn (queries) từ file CSV
- Thực hiện tìm kiếm top 5 kết quả phù hợp nhất cho mỗi query
- Cho phép người dùng đánh giá thủ công số lượng kết quả đúng
- Tính toán độ chính xác (accuracy) cho từng query
- Lưu trữ kết quả để phân tích và đánh giá sau này
- Hỗ trợ tiếp tục công việc đánh giá từ nơi đã dừng

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.7+
- ChromaDB đã được cài đặt và có dữ liệu trong `chromadb_store/`
- File `random_queries.csv` chứa các queries cần đánh giá

### Cài đặt dependencies

```bash
pip install chromadb sentence-transformers
```

## 📁 Cấu trúc Project

```
Chroma_DB_final_data/
├── final_data.py              # Script chính để đánh giá
├── random_queries.csv         # File chứa các queries cần đánh giá
├── chromadb_store/            # Thư mục chứa ChromaDB database
├── progress_final_data.json   # File lưu tiến trình đánh giá (tự động tạo)
├── search_results_data.json   # File lưu kết quả tìm kiếm để đánh giá
└── final_results.json         # File kết quả cuối cùng (tự động tạo khi hoàn thành)
```

## 📖 Hướng dẫn Sử dụng

### 1. Chuẩn bị dữ liệu

Đảm bảo bạn có:
- File `random_queries.csv` với các cột: `query_id`, `query_text`, `category`, `target_person_id`, `difficulty`
- ChromaDB collection `qa_collection` đã được tạo và có dữ liệu

### 2. Chạy chương trình

```bash
python final_data.py
```

### 3. Quy trình đánh giá

1. **Chương trình sẽ hiển thị thông tin query:**
   - Query ID
   - Nội dung query
   - Category và Difficulty
   - Target Person ID

2. **Hiển thị 5 kết quả tìm kiếm:**
   - Mỗi kết quả bao gồm: Title, Skills, Abilities, Program, Distance (độ tương đồng)

3. **Nhập số câu trả lời đúng:**
   - Nhập số từ **0 đến 5** (số lượng kết quả đúng trong top 5)
   - Nhập `exit`, `quit`, hoặc `q` để dừng và lưu tiến trình

4. **Kết quả được lưu tự động:**
   - Sau mỗi query, kết quả được lưu vào `progress_final_data.json`
   - Kết quả tìm kiếm được lưu vào `search_results_data.json`

### 4. Tiếp tục công việc đã dừng

Nếu bạn dừng giữa chừng (bằng cách nhập `exit` hoặc Ctrl+C), chương trình sẽ:
- Tự động lưu tiến trình vào `progress_final_data.json`
- Khi chạy lại, sẽ tự động tiếp tục từ query cuối cùng đã xử lý

**Lưu ý:** Nếu muốn chạy lại từ đầu, xóa file `progress_final_data.json`

### 5. Xử lý theo batch

- Mặc định, mỗi lần chạy sẽ xử lý **20 queries** (có thể thay đổi trong code: `BATCH_SIZE`)
- Sau khi xử lý hết batch, chạy lại script để tiếp tục batch tiếp theo

## 📊 Các File Output

### `progress_final_data.json`
File lưu tiến trình đánh giá, bao gồm:
- `last_processed_index`: Vị trí query cuối cùng đã xử lý
- `results`: Danh sách tất cả queries đã đánh giá với:
  - Thông tin query
  - `correct_count`: Số câu trả lời đúng (0-5)
  - `accuracy`: Độ chính xác (correct_count / 5)
  - `search_results`: 5 kết quả tìm kiếm

**Mục đích:** Cho phép tiếp tục công việc đánh giá từ nơi đã dừng

### `search_results_data.json`
File lưu tất cả kết quả tìm kiếm (không có đánh giá), bao gồm:
- Thông tin query
- `search_results`: 5 kết quả tìm kiếm cho mỗi query

**Mục đích:** Dùng để đánh giá và phân tích kết quả tìm kiếm sau này

### `final_results.json`
File kết quả cuối cùng, được tạo khi xử lý hết tất cả queries:
- Chứa tất cả kết quả đánh giá
- Kèm theo thống kê tổng hợp

## 📈 Thống kê và Báo cáo

Khi hoàn thành tất cả queries, chương trình sẽ hiển thị:
- Tổng số queries đã xử lý
- Độ chính xác trung bình
- Thống kê theo category
- Thống kê theo difficulty level

## ⚙️ Cấu hình

Bạn có thể thay đổi các tham số trong `final_data.py`:

```python
BATCH_SIZE = 20  # Số lượng queries xử lý mỗi lần chạy
COLLECTION_NAME = "qa_collection"  # Tên collection trong ChromaDB
```

Model embedding mặc định: `all-MiniLM-L6-v2` (có thể thay đổi trong code)

## 🔧 Xử lý Lỗi

- **File không tồn tại:** Chương trình sẽ báo lỗi nếu thiếu `random_queries.csv`
- **ChromaDB không kết nối được:** Kiểm tra đường dẫn `chromadb_store/`
- **Dừng giữa chừng:** Nhấn Ctrl+C hoặc nhập `exit` - tiến trình sẽ được lưu tự động

## 📝 Lưu ý

1. **Backup dữ liệu:** Nên backup các file JSON trước khi xóa để tránh mất dữ liệu
2. **Đánh giá nhất quán:** Cố gắng đánh giá theo cùng một tiêu chuẩn để kết quả chính xác
3. **File progress:** File `progress_final_data.json` sẽ tự động bị xóa khi hoàn thành tất cả queries

## 🎯 Tính năng chính

✅ Tìm kiếm semantic với ChromaDB  
✅ Đánh giá thủ công độ chính xác  
✅ Lưu tiến trình để tiếp tục sau  
✅ Lưu kết quả tìm kiếm để phân tích  
✅ Thống kê và báo cáo tự động  
✅ Xử lý theo batch  
✅ Hỗ trợ dừng và tiếp tục an toàn  

## 📞 Hỗ trợ

Nếu gặp vấn đề, kiểm tra:
1. File `random_queries.csv` có đúng format không
2. ChromaDB collection đã được tạo và có dữ liệu chưa
3. Các thư viện đã được cài đặt đầy đủ chưa
