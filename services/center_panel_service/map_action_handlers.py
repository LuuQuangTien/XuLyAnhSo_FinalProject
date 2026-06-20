from image_processing import (
    apply_gray,
    apply_gaussian_filter,
    apply_sobel,
    apply_prewitt,
)

ACTION_MAP = {
    'grayscale': apply_gray.process,
    'gaussian_filter': apply_gaussian_filter.process,
    'sobel_filter': apply_sobel.process,
    'prewitt_filter': apply_prewitt.process,
}

def get_handler(action_name):
    return ACTION_MAP.get(action_name)
