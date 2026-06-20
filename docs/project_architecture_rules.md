# Kiến trúc Dự án & Quy tắc Lập trình (SOLID & Pure DAG Node Engine)

Tài liệu này mô tả cấu trúc hiện tại của dự án OMR và đưa ra các quy tắc (rules) cốt lõi nhằm đảm bảo mã nguồn tuân thủ các nguyên lý thiết kế phần mềm, dễ bảo trì, và không phá vỡ kiến trúc Node Engine đặc biệt của dự án.

## 1. Tổng quan Kiến trúc hiện tại

Dự án đang kết hợp mô hình **MVC/MVP** cho phần Giao diện (UI) và kiến trúc **Pure DAG Node Engine** cho phần Xử lý Cốt lõi (Core Engine):

- **`ui/` (View):** Chứa thuần túy mã giao diện người dùng (PyQt6). 
- **`controllers/` (Controller/Presenter):** Cầu nối giao tiếp giữa Giao diện và Nghiệp vụ. Lắng nghe các sự kiện (signals) từ giao diện, gọi Services/Workers để xử lý ở luồng nền (background threads), sau đó cập nhật lại giao diện.
- **`services/` (Model/Business Logic):** Chứa logic nghiệp vụ, quản lý trạng thái (`ImageStateManager`), xử lý cache. Đặc biệt là `omr_service.py` đóng vai trò là Lõi khởi chạy DAG Engine.
- **`image_processing/` (Computer Vision):** Thư viện độc lập chỉ cung cấp các hàm thuật toán (Alignment, Thresholding, Extraction). Không hề biết về luồng đi của dữ liệu.
- **`services/grading_service/nodes/` (Graph Nodes):** Đóng gói các hàm OpenCV thuần túy thành các "Node" tuân thủ tiêu chuẩn Đầu vào - Đầu ra. Đây là trái tim của hệ thống OMR.

---

## 2. Các Nguyên tắc Thiết kế (SOLID)

### S - Single Responsibility Principle (Đơn Trách nhiệm)
- **Trong UI/Controllers:** Mỗi Controller quản lý một mảng riêng biệt (`ThemeController`, `FileIOController`).
- **Trong Node Engine:** Mỗi Node CHỈ LÀM MỘT VIỆC. Tuyệt đối cấm viết một Node ôm đồm nhiều việc. 
  - *Ví dụ:* Node đọc bong bóng (`BubbleGridDetector`) chỉ đếm pixel. Node chấm điểm (`MCQScorer`) chỉ so sánh mảng. Node vẽ đồ họa (`MCQVisualizer`) chỉ cầm cọ vẽ. Không trộn lẫn CV (Computer Vision) với Logic kinh doanh.

### O - Open/Closed Principle (Đóng/Mở)
- **Thực trạng Pipeline:** Nhờ có **Data-Driven Pure DAG Engine**, nếu muốn tạo một mẫu phiếu thi mới (đổi số câu, vị trí, lưới bong bóng), ta **KHÔNG CẦN VIẾT THÊM CODE PYTHON**. Chỉ việc tạo một file cấu hình JSON mới. Hệ thống mở cho việc mở rộng (thêm template) nhưng đóng với việc sửa đổi mã nguồn.

### I & D - Interface Segregation & Dependency Inversion
- `omr_service.py` không gọi thẳng các thuật toán xử lý ảnh. Nó phụ thuộc vào một Registry trừu tượng (`NODE_CLASS_MAPPINGS`) và chỉ thực thi các hàm thông qua interface `execute(**kwargs) -> dict`.

---

## 3. Quy tắc cốt lõi khi Thêm/Sửa Code (BẮT BUỘC TUÂN THỦ)

Hãy coi đây là **Checklist** mỗi khi bạn viết code mới:

### Rule 1: Kiến trúc Pure DAG Node Engine là Bất khả xâm phạm
- Mọi quy trình xử lý ảnh và chấm điểm **bắt buộc** phải được thiết kế thành các Node độc lập trong thư mục `services/grading_service/nodes/`.
- File cấu hình JSON (templates) là nơi duy nhất quyết định luồng đi của dữ liệu. Cấu trúc liên kết của JSON dùng biến tham chiếu: `@node_id.output_key`. 
- **Tuyệt đối cấm:** Hardcode thứ tự chạy các hàm thuật toán bên trong `omr_service.py`. `omr_service.py` chỉ được quyền lặp qua đồ thị JSON và nối dây.

### Rule 2: KHÔNG viết Business Logic vào thư mục `ui/`
- Tương tự, UI **chỉ được phép** tạo widget và phát `pyqtSignal`. Cấm đọc file, xử lý ảnh, hay thay đổi trạng thái gốc bên trong các class UI.

### Rule 3: Controller chỉ là "Người Điều phối" (Mediator)
- Controller làm nhiệm vụ: `Lắng nghe UI -> Gọi Service/Worker -> Cập nhật lại UI`. Nếu logic xử lý ảnh quá dài, hãy đóng gói nó vào Node và gọi thông qua Engine.

### Rule 4: Tách luồng (Threading) cho tác vụ nặng
- Tránh làm đơ giao diện PyQt6. Mọi tác vụ đọc ghi file, chấm thi hàng loạt đều phải ném vào `QThread` hoặc `QRunnable` (trong `controllers/workers/`).

### Rule 5: Nguyên tắc "Tell, Don't Ask"
- Khi Controller thao tác với View, hãy yêu cầu View tự thay đổi (gọi hàm `set_xyz()`), thay vì Controller truy cập vào tận bên trong các biến của View để tự chỉnh sửa.

### Rule 6: Giữ cấu trúc Import sạch sẽ
- Các file ở lớp Data/Logic (`services/`, `nodes/`, `image_processing/`) **tuyệt đối không được** import các tệp từ `ui/` hay `controllers/`. Data layer không được phép biết về Presentation layer. Luôn dùng Absolute Import.

## Tổng kết Quy trình khi tạo Mẫu (Template) mới:
1. Bạn **KHÔNG** cần đụng vào code Python.
2. Tạo file JSON ở thư mục `assets/templates/`.
3. Định nghĩa mảng `"nodes"` phẳng. Chọn các `type` Node phù hợp có sẵn trong Registry.
4. Cắm dây dữ liệu (Link) từ Node này sang Node kia bằng biểu thức `"inputs": {"tham_số": "@tên_node_trước.tên_output"}`.
5. Khởi động lại ứng dụng. Engine sẽ tự động render và biên dịch đồ thị của bạn!
