# Phân tích Luồng Hoạt động (Workflows) của Hệ thống OMR

Tài liệu này mô tả chi tiết các luồng hoạt động (flows) chính trong ứng dụng, từ quá trình khởi động, tương tác cơ bản đến nghiệp vụ chấm thi phức tạp. Ứng dụng này theo cấu trúc phân tầng rõ rệt (View - Controller - Service).

---

## 1. Luồng Khởi động Ứng dụng (Startup Flow)
Đây là quy trình đầu tiên khi ứng dụng được kích hoạt (từ file `main.py`).

1. **Khởi tạo Ứng dụng:** Set AppID để hiển thị đúng icon dưới thanh Taskbar (Windows) và khởi tạo `QApplication`.
2. **Nạp Stylesheet:** Load bộ màu CSS chung toàn cục từ `ui.styles.GLOBAL_STYLE`.
3. **Hiển thị Intro:** Bật hộp thoại `GroupInfoDialog` (Giới thiệu thành viên nhóm). Người dùng nhấn "Vào Ứng Dụng" thì hộp thoại mới đóng.
4. **Khởi tạo UI:** Khởi tạo `MainWindow` (Cửa sổ chính) cùng các panel rỗng (`LeftPanel`, `CenterPanel`, `RightPanel`, `AppToolBar`).
5. **Liên kết Logic (Wiring):** Khởi tạo `MainController` và truyền `MainWindow` vào. Tại đây, hệ thống sinh ra `ImageStateManager` (dùng để lưu trữ dữ liệu ảnh dùng chung) và các Sub-Controllers khác (`Theme`, `UIState`, `FileIO`, `Explorer`, `ImageProcessing`, `OMR`). Cáp (bind) các tín hiệu sự kiện của UI vào các Controller tương ứng.
6. **Hiển thị:** Show `MainWindow` và chạy vòng lặp sự kiện `app.exec()`.

---

## 2. Luồng Quản lý Tệp và Hiển thị (Explorer & Workspace Flow)
Quản lý việc duyệt thư mục và mở ảnh ở tab giữa. Điều phối bởi `ExplorerController`.

1. **Nhập Thư mục (Import Folder):**
   - Click nút trên Toolbar -> `ExplorerController` lấy đường dẫn thư mục.
   - Quét tìm tất cả các file hợp lệ (png, jpg...).
   - Đổ danh sách thư mục và file vào cây (TreeWidget) bên `LeftPanel`.
2. **Mở Ảnh (Open Image):**
   - Người dùng click đúp vào một ảnh bên trái. `LeftPanel` phát tín hiệu `image_selected`.
   - `ExplorerController` bắt tín hiệu, gọi lệnh tới `CenterPanel` yêu cầu tạo/chuyển một Tab mới để chứa ảnh.
   - Khởi tạo một phiên ảnh (session) trong `ImageStateManager` (lưu trữ ma trận gốc, ma trận đang làm việc, độ zoom).
   - `UIStateController` sẽ đánh thức (enable) các nút công cụ chức năng (như Crop, Chấm thi) do lúc này đã có hình ảnh sẵn sàng.

---

## 3. Luồng Xử lý Ảnh Thủ công (Manual Image Processing Flow)
Khi người dùng muốn tự tay điều chỉnh 1 bức ảnh. Điều phối bởi `ImageProcessingController`.

1. **Kích hoạt công cụ:** Người dùng bấm một tính năng trên Toolbar (ví dụ: Nắn thẳng / Tăng độ tương phản).
2. **Chuyển giao Logic:** Tín hiệu truyền vào `ImageProcessingController`. Tại đây, ứng dụng lấy ma trận ảnh hiện tại từ `ImageStateManager`.
3. **Xử lý Core:** Controller gọi xuống tầng `services` hoặc gọi thẳng thuật toán bên `image_processing/`.
4. **Cập nhật UI:**
   - Ảnh mới được lưu ngược lại `ImageStateManager`. Cờ `is_modified` bật thành `True`.
   - Lệnh render được đẩy xuống `CenterPanel` để người dùng thấy ngay sự thay đổi.

---

## 4. Luồng Chấm Thi Trắc Nghiệm Tự Động (Data-Driven OMR Grading Flow)
Luồng phức tạp nhất trong ứng dụng, chịu trách nhiệm chấm điểm hàng loạt. Hệ thống hoạt động như một Engine dựa trên cấu trúc Node Pipeline linh hoạt. Điều phối bởi `OMRController` và `OMRBatchWorker`.

1. **Thiết lập Cấu hình (Setup):**
   - Người dùng chọn "Chấm thi lô". Bật hộp thoại `BatchSetupDialog`.
   - Nhập thư mục đầu vào, thư mục đầu ra, file Excel cấu hình đáp án và lựa chọn Mẫu phiếu (Template). Các mẫu này được tự động load từ các file cấu hình `.json` trong thư mục `assets/templates/`.
2. **Bước Tiền xử lý (Pre-scan Phase):**
   - Khóa toàn bộ giao diện (chặn tương tác để tránh lỗi).
   - Khởi chạy một tiểu trình (Thread: `OMRBatchWorker`) ở chế độ `'prescan'`.
   - **Mục đích:** `OMRService` đóng vai trò là Pipeline Executor, chạy một vòng kiểm tra cực nhanh trên tập ảnh dựa vào cấu trúc JSON để phát hiện ảnh mờ, mất điểm neo. Nó trích xuất nhanh Số báo danh và Mã đề.
   - **Báo cáo Tiền kiểm:** Hiển thị `PreScanReportDialog` tổng kết số ảnh đủ chuẩn và số ảnh lỗi. Yêu cầu người dùng xác nhận đi tiếp.
3. **Bước Chấm điểm Chính thức (Pipeline Execution Phase):**
   - Chạy Thread `OMRBatchWorker` ở chế độ `'grade'` trên những ảnh hợp lệ.
   - Gọi hàm `OMRService.grade_image()`. Tại đây, Data-Driven Pipeline bắt đầu chạy:
     - **Tầng Image Processing:** Dựa theo JSON, hệ thống bật/tắt module Alignment, cấu hình hàm Threshold và gọi Extractor lấy ra các khối chữ nhật (Blocks).
     - **Tầng Business Logic (Nodes):** `OMRService` kiểm tra `"regions"` trong JSON và gọi các Node Processor tương ứng (ví dụ: `sbd_reader`, `multi_col_reader`) nằm tại `services/grading_service/nodes/`. Mỗi Node sẽ chịu trách nhiệm chấm điểm, so đáp án Excel và tạo thông báo lỗi nếu có.
   - Bản sao của ảnh được vẽ khoanh đỏ/xanh (đúng/sai) và lưu vào thư mục đầu ra.
4. **Xuất Báo cáo & Thống kê (Reporting Phase):**
   - Tiến trình kết thúc, `ReportExportService` xuất danh sách tổng điểm ra tệp Excel `tong_hop_diem.xlsx`.
   - Cập nhật số liệu (tỷ lệ Giỏi, Khá, Trung bình, Yếu, biểu đồ tỷ lệ) sang `RightPanel` Dashboard.
   - Tự động load thư mục Output vào `LeftPanel` để người dùng có thể duyệt ngay các ảnh đã chấm xong. Mở khóa giao diện.

---

## 5. Luồng Quản lý Vòng Đời và Dọn dẹp (Lifecycle & Cleanup Flow)
Kiểm soát trạng thái khi người dùng tắt các Tab hoặc đóng toàn bộ chương trình.

1. **Đóng Tab (Tab Closed):**
   - Gọi hàm `MainController.handle_tab_close`.
   - Kiểm tra `is_modified` trong State Manager. Nếu True, bung Dialog hỏi "Lưu thay đổi không?".
   - Cập nhật lại UI state (Nếu hết tab, phải khóa (disable) các nút công cụ).
2. **Thoát Ứng dụng (App Close):**
   - `MainWindow.closeEvent` kích hoạt.
   - Nếu có nhiều ảnh chưa lưu, gom chung vào `ask_save_all_exit`. 
   - Trước khi chương trình thực sự ngắt hoàn toàn, `CacheManager.clear_all()` sẽ được gọi để dọn dẹp các ảnh nháp `.tmp` hay thumbnail giải phóng bộ nhớ.
