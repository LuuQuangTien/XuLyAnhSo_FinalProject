from abc import ABC, abstractmethod
import numpy as np

class BaseOMRTemplate(ABC):
    # Các thuộc tính dùng cho UI
    name: str = "Base Template"
    description: str = "Mô tả mẫu"
    preview_image_path: str = "" # Đường dẫn tới ảnh xem trước nhỏ gọn

    @abstractmethod
    def calculate_confidence(self, blocks: list, image_shape: tuple = None) -> float:
        """
        Tính toán độ tin cậy của bức ảnh với mẫu này.
        Trả về float từ 0.0 đến 1.0
        """
        pass

    @abstractmethod
    def grade(self, image: np.ndarray, thresh: np.ndarray, blocks: list, answers: dict) -> tuple[np.ndarray, str, dict]:
        """
        Thực hiện logic chấm điểm dựa trên lưới tọa độ đặc thù của mẫu.
        Trả về (ảnh_đã_chấm, chuỗi_kết_quả, dict_điểm_số)
        """
        pass
