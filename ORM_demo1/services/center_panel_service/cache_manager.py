# Lưu từng bước ảnh (numpy) ra thư mục Temp của Windows — Đợt 1: chỉ quản lý file.
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR_NAME = "ImageEditorCache"


def _session_dir_name(tab_id: str) -> str:
    digest = hashlib.sha256(tab_id.encode("utf-8")).hexdigest()[:16]
    return f"Tab_{digest}"


class CacheManager:
    """Ghi/đọc/xóa các file step_k.npy theo tab_id (đường dẫn file gốc)."""

    def __init__(self) -> None:
        self._root: Optional[Path] = None

    def init_cache(self) -> None:
        """Đặt thư mục gốc dưới %TEMP%/ImageEditorCache và xóa sạch nội dung cũ (crash lần trước)."""
        root = Path(tempfile.gettempdir()) / ROOT_DIR_NAME
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root

    def _require_root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Gọi CacheManager.init_cache() trước khi dùng cache.")
        return self._root

    def _session_path(self, tab_id: str) -> Path:
        return self._require_root() / _session_dir_name(tab_id)

    def create_tab_session(self, tab_id: str) -> None:
        self._session_path(tab_id).mkdir(parents=True, exist_ok=True)

    def save_step(self, tab_id: str, step_index: int, image_data: np.ndarray) -> None:
        session = self._session_path(tab_id)
        session.mkdir(parents=True, exist_ok=True)
        path = session / f"step_{step_index}.npy"
        np.save(path, image_data, allow_pickle=False)

    def load_step(self, tab_id: str, step_index: int) -> np.ndarray:
        path = self._session_path(tab_id) / f"step_{step_index}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Không có file lịch sử: {path}")
        data = np.load(path, allow_pickle=False)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Dữ liệu không phải ndarray: {path}")
        return data

    def delete_future_steps(self, tab_id: str, from_index: int) -> None:
        """Xóa step_{from_index}.npy trở đi (khi undo rồi chỉnh sửa nhánh mới)."""
        session = self._session_path(tab_id)
        if not session.is_dir():
            return
        for path in session.glob("step_*.npy"):
            try:
                suffix = path.stem.split("_", 1)[1]
                idx = int(suffix)
            except (ValueError, IndexError):
                continue
            if idx >= from_index:
                path.unlink(missing_ok=True)

    def clear_tab_cache(self, tab_id: str) -> None:
        p = self._session_path(tab_id)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    def clear_all(self) -> None:
        """Xóa toàn bộ cache; giữ lại thư mục gốc rỗng nếu vẫn còn phiên làm việc trong process."""
        root = self._root
        if root is None:
            return
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)


cache_manager = CacheManager()
