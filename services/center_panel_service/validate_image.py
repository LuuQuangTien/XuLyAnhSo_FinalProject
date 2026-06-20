# Kiểm tra xem ảnh đã được tải lên thành công hay chưa.
from services.notification_service import warn_no_image_selected

def is_image_loaded(image_state, parent=None, show_warning=True):
    """
    Checks if an image is currently loaded in the state.
    
    Args:
        image_state: The ImageState instance to check.
        parent: The UI parent for the warning dialog.
        show_warning: Whether to show a warning dialog if no image is loaded.
        
    Returns:
        True if an image is loaded, False otherwise.
    """
    if not image_state or not image_state.has_image():
        if show_warning:
            warn_no_image_selected(parent)
        return False
    return True
