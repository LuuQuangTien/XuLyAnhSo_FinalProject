# Hàm điều phối việc gọi thuật toán xử lý ảnh từ Action Registry.
from services.center_panel_service.map_action_handlers import get_handler

def execute(action_name, current_image, **kwargs):
    """
    Coordinates the execution of an image processing action.
    
    Args:
        action_name: The identifier of the action to run.
        current_image: The image (numpy array) to process.
        **kwargs: Additional parameters for the action (e.g., channel_index).
        
    Returns:
        The processed image, or None if the action failed or was not found.
    """
    handler = get_handler(action_name)
    
    if not handler:
        print(f"Action not found: {action_name}")
        return None
        
    try:
        processed_image = handler(current_image, **kwargs)
        return processed_image
    except Exception as e:
        print(f"Error executing action {action_name}: {e}")
        return None
