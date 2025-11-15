# 5. Kết luận

## 5.1. Đánh giá kết quả

### 5.1.1. Tổng quan

Hệ thống đã được đánh giá trên **140 queries** với các metrics chính:

- **Precision@5 trung bình**: 0.9029 (90.29%)
- **MAP@5 (Mean Average Precision@5)**: 0.9714 (97.14%)

### 5.1.2. Phân tích chi tiết Precision@5

- **Min**: 0.0000
- **Max**: 1.0000
- **Median**: 1.0000

**Phân bố Precision@5:**
- Perfect (1.0000): 120 queries (85.7%)
- High (0.80-0.99): 2 queries (1.4%)
- Medium (0.50-0.79): 3 queries (2.1%)
- Low (<0.50): 15 queries (10.7%)

### 5.1.3. Phân tích số lượng kết quả relevant

- Tổng số kết quả relevant: 632/700
- Tỷ lệ relevant: 90.29%
- Số lượng relevant trung bình mỗi query: 4.51/5

### 5.1.4. Thống kê theo Category

- **BE** (n=32):
  - Precision@5: 0.9375 (min: 0.0000, max: 1.0000, perfect: 29)
  - AP@5: 0.9688

- **CRM** (n=1):
  - Precision@5: 0.2000 (min: 0.2000, max: 0.2000, perfect: 0)
  - AP@5: 1.0000

- **DBA** (n=6):
  - Precision@5: 0.8333 (min: 0.0000, max: 1.0000, perfect: 5)
  - AP@5: 0.8333

- **FE** (n=12):
  - Precision@5: 1.0000 (min: 1.0000, max: 1.0000, perfect: 12)
  - AP@5: 1.0000

- **IT** (n=3):
  - Precision@5: 1.0000 (min: 1.0000, max: 1.0000, perfect: 3)
  - AP@5: 1.0000

- **NETSEC** (n=27):
  - Precision@5: 0.9778 (min: 0.4000, max: 1.0000, perfect: 26)
  - AP@5: 1.0000

- **OTHER** (n=49):
  - Precision@5: 0.8122 (min: 0.0000, max: 1.0000, perfect: 35)
  - AP@5: 0.9592

- **PM** (n=10):
  - Precision@5: 1.0000 (min: 1.0000, max: 1.0000, perfect: 10)
  - AP@5: 1.0000

### 5.1.5. Thống kê theo Difficulty

- **hard** (n=26):
  - Precision@5: 0.8385 (min: 0.0000, max: 1.0000, perfect: 20)
  - AP@5: 0.9615

- **standard** (n=114):
  - Precision@5: 0.9175 (min: 0.0000, max: 1.0000, perfect: 100)
  - AP@5: 0.9737

### 5.1.6. Phân tích Distance

- Distance trung bình: 0.6135
- Min: 0.3154 | Max: 1.0533 | Median: 0.6004

### 5.1.7. Đánh giá chất lượng hệ thống

**Kết luận**: Hệ thống hoạt động **RẤT TỐT** (Precision@5 = 0.9029).

Hệ thống đạt độ chính xác rất cao với Precision@5 >= 90%, cho thấy khả năng tìm kiếm semantic hoạt động hiệu quả.

Với MAP@5 = 0.9714, hệ thống cho thấy khả năng xếp hạng các kết quả relevant ở vị trí cao trong danh sách kết quả tìm kiếm.

## 5.2. Hạn chế

### 5.2.1. Hạn chế về phương pháp đánh giá

1. **Đánh giá dựa trên distance threshold**: 
   - Hệ thống hiện tại sử dụng distance threshold = 0.8 để xác định kết quả relevant
   - Threshold này có thể quá cao, dẫn đến hầu hết các kết quả top 5 đều được coi là relevant
   - Điều này có thể làm cho Precision@5 và MAP@5 cao hơn so với đánh giá thực tế

2. **Thiếu đánh giá thủ công**:
   - Kết quả được đánh giá tự động dựa trên distance, không có sự xác nhận từ người dùng
   - Có thể có các kết quả có distance thấp nhưng không thực sự phù hợp với query

3. **Không sử dụng target_person_id**:
   - Mặc dù có thông tin target_person_id trong dữ liệu, nhưng không được sử dụng để đánh giá
   - Điều này có thể dẫn đến việc đánh giá không phản ánh đúng mục tiêu tìm kiếm

### 5.2.2. Hạn chế về dữ liệu

1. **Phân bố không đều theo category**:
   - Một số category có số lượng queries ít (ví dụ: CRM chỉ có 1 query, IT có 3 queries)
   - Điều này làm cho thống kê theo category không đại diện đầy đủ

2. **Phân bố không đều theo difficulty**:
   - Số lượng queries "standard" (114) nhiều hơn đáng kể so với "hard" (26)
   - Có thể cần thêm dữ liệu để đánh giá chính xác hơn về độ khó

### 5.2.3. Hạn chế về kỹ thuật

1. **Model embedding**:
   - Sử dụng model `all-MiniLM-L6-v2` - một model nhỏ và nhanh nhưng có thể không tối ưu cho tiếng Việt hoặc domain cụ thể
   - Có thể cải thiện bằng cách sử dụng model lớn hơn hoặc fine-tune trên dữ liệu domain

2. **Chỉ tìm kiếm top 5**:
   - Hệ thống chỉ trả về 5 kết quả đầu tiên
   - Có thể bỏ sót các kết quả relevant ở vị trí thấp hơn

3. **Không có re-ranking**:
   - Kết quả được sắp xếp chỉ dựa trên distance từ ChromaDB
   - Không có bước re-ranking để cải thiện thứ tự kết quả

### 5.2.4. Hạn chế về metrics

1. **Chỉ đánh giá Precision@K và MAP@K**:
   - Thiếu các metrics khác như Recall@K, NDCG@K
   - Không đánh giá về thời gian phản hồi hoặc hiệu suất hệ thống

2. **Không có đánh giá theo từng vị trí**:
   - Chỉ đánh giá tổng thể top 5, không phân tích chi tiết từng vị trí (vị trí 1, 2, 3, 4, 5)
   - Không biết được liệu các kết quả relevant có được xếp ở vị trí cao hay không

### 5.2.5. Đề xuất cải thiện

1. **Điều chỉnh threshold**:
   - Giảm distance threshold xuống 0.6-0.7 để đánh giá chặt chẽ hơn
   - Hoặc sử dụng relevance score kết hợp distance và keyword matching

2. **Thêm đánh giá thủ công**:
   - Cho phép người dùng đánh giá thủ công một phần queries để validate kết quả tự động
   - So sánh kết quả đánh giá tự động và thủ công

3. **Cải thiện model**:
   - Thử nghiệm với các model embedding lớn hơn hoặc chuyên biệt hơn
   - Fine-tune model trên dữ liệu domain nếu có thể

4. **Mở rộng metrics**:
   - Thêm Recall@K, NDCG@K để đánh giá toàn diện hơn
   - Phân tích chi tiết theo từng vị trí trong top K

5. **Tăng số lượng kết quả**:
   - Tăng từ top 5 lên top 10 hoặc top 20 để đánh giá recall tốt hơn
   - Thêm bước re-ranking để cải thiện thứ tự kết quả
