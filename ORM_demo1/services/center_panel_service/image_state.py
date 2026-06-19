# Quản lý danh sách các ảnh đang mở trên nhiều Tab + lịch sử chỉnh sửa trên đĩa (Đợt 2).
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from services.center_panel_service.cache_manager import cache_manager

# step_0 = ảnh gốc khi load; cho phép thêm tối đa 50 bước (step_1 … step_50).
MAX_STEP_INDEX = 50


@dataclass
class ImageState:
    tab_id: str
    current_image_path: Optional[str] = None
    current_output_path: Optional[str] = None
    current_processed_image: Optional[np.ndarray] = None
    is_modified: bool = False
    history_index: int = 0
    history_count: int = 0
    history_save_index: int = 0
    
    # Lưu giá trị thanh trượt riêng cho từng ảnh
    log_c: int = 1
    gamma_val: int = 10 # Giá trị slider (10 = 1.0)
    piecewise_r1: int = 50
    piecewise_r2: int = 200
    kernel_size: int = 3 # Giá trị mặc định cho các bộ lọc
    freq_d0: int = 30  # Cut-off frequency cho bộ lọc miền tần số
    freq_order_n: int = 2  # Bậc Butterworth

    def has_image(self) -> bool:
        return self.current_processed_image is not None

    def set_loaded_image(self, image_path: str, image_data: np.ndarray) -> bool:
        try:
            cache_manager.create_tab_session(self.tab_id)
            cache_manager.save_step(self.tab_id, 0, image_data)
        except OSError:
            return False

        self.current_image_path = image_path
        self.current_output_path = image_path
        self.current_processed_image = image_data
        self.is_modified = False
        self.history_index = 0
        self.history_count = 1
        self.history_save_index = 0
        return True

    def set_processed_image(self, image_data: np.ndarray) -> bool:
        new_idx = self.history_index + 1
        if new_idx > MAX_STEP_INDEX:
            return False

        if self.history_index < self.history_count - 1:
            cache_manager.delete_future_steps(self.tab_id, self.history_index + 1)
            self.history_count = self.history_index + 1

        try:
            cache_manager.save_step(self.tab_id, new_idx, image_data)
        except OSError:
            return False

        self.history_index = new_idx
        self.history_count = new_idx + 1
        self.current_processed_image = image_data
        self.is_modified = True
        return True

    def set_output_path(self, output_path: str) -> None:
        self.current_output_path = output_path
        self.is_modified = False
        self.history_save_index = self.history_index

    def can_undo(self) -> bool:
        return self.history_index > 0

    def can_redo(self) -> bool:
        return self.history_index < self.history_count - 1

    def undo(self) -> bool:
        if not self.can_undo():
            return False
        self.history_index -= 1
        self.current_processed_image = cache_manager.load_step(self.tab_id, self.history_index)
        self.is_modified = self.history_index != self.history_save_index
        return True

    def redo(self) -> bool:
        if not self.can_redo():
            return False
        self.history_index += 1
        self.current_processed_image = cache_manager.load_step(self.tab_id, self.history_index)
        self.is_modified = self.history_index != self.history_save_index
        return True

    def reset(self) -> bool:
        if not self.has_image():
            return False
        cache_manager.delete_future_steps(self.tab_id, 1)
        self.current_processed_image = cache_manager.load_step(self.tab_id, 0)
        self.history_index = 0
        self.history_count = 1
        self.history_save_index = 0
        self.is_modified = False
        return True


class ImageStateManager:
    """Manages the state of multiple images across different tabs."""

    def __init__(self) -> None:
        self.states: Dict[str, ImageState] = {}  # tab_id (file path) -> ImageState
        self.active_tab_id: Optional[str] = None

    def get_active_state(self) -> Optional[ImageState]:
        if self.active_tab_id and self.active_tab_id in self.states:
            return self.states[self.active_tab_id]
        return None

    def create_or_get_state(self, tab_id: str) -> ImageState:
        if tab_id not in self.states:
            self.states[tab_id] = ImageState(tab_id=tab_id)
        return self.states[tab_id]

    def remove_state(self, tab_id: str) -> None:
        if tab_id in self.states:
            cache_manager.clear_tab_cache(tab_id)
            del self.states[tab_id]
            if self.active_tab_id == tab_id:
                self.active_tab_id = None

    def set_active_tab(self, tab_id: str) -> None:
        self.active_tab_id = tab_id

    def get_unsaved_count(self) -> int:
        return sum(1 for state in self.states.values() if state.is_modified)

    def get_all_unsaved_states(self) -> Dict[str, ImageState]:
        return {tid: st for tid, st in self.states.items() if st.is_modified}

    def has_any_image(self) -> bool:
        return any(state.has_image() for state in self.states.values())
