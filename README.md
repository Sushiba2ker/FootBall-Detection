# Football Analysis Project

Một hệ thống phân tích video bóng đá sử dụng AI để phát hiện, theo dõi cầu thủ và phân tích chiến thuật trận đấu.

## Tính năng chính

- **Phát hiện đối tượng**: Phát hiện cầu thủ, thủ môn, trọng tài và bóng trong video
- **Theo dõi đối tượng**: Duy trì ID nhất quán cho các cầu thủ qua các khung hình
- **Phân loại đội**: Tự động phân biệt hai đội dựa trên màu sắc áo đấu
- **Phân tích kiểm soát bóng**: Tính toán phần trăm kiểm soát bóng của mỗi đội
- **Thống kê chuyển động**: Theo dõi khoảng cách di chuyển và vùng hoạt động của cầu thủ
- **Phân tích đội hình**: Theo dõi vị trí và chiến thuật của đội

## Cấu trúc dự án

```
FootBall-Detection/
├── main.py                 # Tệp chính để chạy phân tích
├── config.py              # Cấu hình và hằng số
├── video_processor.py     # Xử lý video và phát hiện đối tượng
├── analysis.py            # Phân tích kiểm soát bóng và thống kê
├── requirements.txt       # Danh sách thư viện cần thiết
├── TODO.md               # Kế hoạch cải tiến dự án
├── README.md             # Tài liệu hướng dẫn (tệp này)
└── Scout_Football.ipynb  # Notebook gốc (để tham khảo)
```

## Cài đặt

### 1. Cài đặt thư viện cơ bản

```bash
pip install -r requirements.txt
```

### 2. Cài đặt thư viện Sports từ Roboflow

```bash
pip install git+https://github.com/roboflow/sports.git
```

### 3. Thiết lập API Key

Cần có Roboflow API key để sử dụng các mô hình phát hiện:

#### Trên Google Colab:

```python
from google.colab import userdata
# Thêm ROBOFLOW_API_KEY vào Colab Secrets
```

#### Trên máy tính cá nhân:

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

## Sử dụng

### Chạy phân tích cơ bản

```python
python main.py
```

### Tùy chỉnh đường dẫn video

Chỉnh sửa tệp `config.py`:

```python
SOURCE_VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_VIDEO_PATH = "path/to/output/video.mp4"
```

### Sử dụng các module riêng lẻ

#### Xử lý video

```python
from video_processor import VideoProcessor
from config import *

# Khởi tạo processor
processor = VideoProcessor(detection_model)

# Xử lý video
processor.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    team_classifier=team_classifier
)
```

#### Phân tích kiểm soát bóng

```python
from analysis import BallPossessionAnalyzer

analyzer = BallPossessionAnalyzer()

# Cập nhật trong vòng lặp xử lý frame
possession = analyzer.update_possession(
    ball_detections, player_detections, player_team_ids, frame_number
)

# Lấy thống kê cuối cùng
percentages = analyzer.get_possession_percentages(total_frames)
```

## Cấu hình

### Tham số chính trong `config.py`

- `CONFIDENCE_THRESHOLD`: Ngưỡng tin cậy cho phát hiện đối tượng (mặc định: 0.3)
- `NMS_THRESHOLD`: Ngưỡng Non-Maximum Suppression (mặc định: 0.5)
- `TEAM_CLASSIFICATION_INTERVAL`: Khoảng cách frame để phân loại lại đội (mặc định: 30)
- `BALL_CONTROL_DISTANCE_THRESHOLD`: Khoảng cách để xác định kiểm soát bóng (mặc định: 50)

### Màu sắc

Các màu sắc cho đội và đối tượng được định nghĩa trong `COLORS` dictionary:

```python
COLORS = {
    'team_0_players': '#00BFFF',      # Xanh cho đội 0
    'team_1_players': '#FF1493',      # Hồng cho đội 1
    'team_0_goalkeeper': '#32CD32',   # Xanh lá cho thủ môn đội 0
    'team_1_goalkeeper': '#FF0000',   # Đỏ cho thủ môn đội 1
    'referee': '#000000',             # Đen cho trọng tài
    'ball': '#FFD700',                # Vàng cho bóng
}
```

## Output

Dự án tạo ra:

1. **Video đã chú thích**: Video với bounding box, ID cầu thủ và thông tin kiểm soát bóng
2. **Log file**: `football_analysis.log` chứa thông tin chi tiết về quá trình xử lý
3. **Thống kê cuối cùng**: In ra console và log file

### Ví dụ output thống kê:

```
Team 0 possession: 52.3%
Team 1 possession: 47.7%
Total possession switches: 23
```

## Tối ưu hóa hiệu suất

### GPU

Dự án được tối ưu cho GPU. Đảm bảo có:

- CUDA được cài đặt
- PyTorch với CUDA support
- ONNX Runtime GPU

### Bộ nhớ

Để xử lý video lớn:

- Giảm `BATCH_SIZE` trong config nếu thiếu GPU memory
- Tăng `STRIDE` để xử lý ít frame hơn khi training team classifier

## Troubleshooting

### Lỗi thường gặp

1. **"Roboflow API key not found"**

   - Kiểm tra API key đã được set chính xác
   - Đảm bảo có quyền truy cập model

2. **"Could not import sports library"**

   ```bash
   pip install git+https://github.com/roboflow/sports.git
   ```

3. **CUDA errors**

   - Kiểm tra CUDA compatibility
   - Đổi `DEVICE = "cpu"` trong config.py nếu cần

4. **Memory errors**
   - Giảm batch size
   - Xử lý video ngắn hơn để test

### Logging

Kiểm tra file log `football_analysis.log` để debug:

```bash
tail -f football_analysis.log
```

## Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License

Dự án sử dụng các thư viện mã nguồn mở. Xem tệp requirements.txt để biết chi tiết.

## Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub repository.
