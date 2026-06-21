import cv2
import os
from image_processing.alignment.document_aligner import align_document
from image_processing.preprocessing.apply_adaptive_threshold import process as apply_adaptive_threshold
from image_processing.extraction.block_extractor import find_blocks

class ImageStandardizerNode:
    def execute(self, image, min_width=1400, max_width=2000):
        h, w = image.shape[:2]
        if w < min_width:
            scale = min_width / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        elif w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return {"image": image}

class DocumentAlignerNode:
    def execute(self, image, method="four_corners", debug_dir=None, debug_prefix=""):
        aligned, error_msg = align_document(image, debug_dir, debug_prefix, method=method)
        return {"image": aligned, "error": error_msg}

class AdaptiveThresholdNode:
    def execute(self, image, block_size=None, dynamic_ratio=45, C=10, sharpen=True):
        if block_size is None:
            w = image.shape[1]
            block_size = int(w / dynamic_ratio)
            if block_size % 2 == 0:
                block_size += 1
            if block_size < 11:
                block_size = 11
                
        # Thresh 1: Dùng để tìm khối (Block size nhỏ)
        thresh = apply_adaptive_threshold(image, block_size=block_size, C=C, sharpen=sharpen, blur=True)
        
        # Thresh 2: Chuyên dụng cho đọc nét chì bong bóng (Block size rất lớn để chống "rỗng ruột")
        bubble_thresh = apply_adaptive_threshold(image, block_size=91, C=10, sharpen=False, blur=False)
        
        return {"image": thresh, "bubble_thresh": bubble_thresh}

class BlockExtractorNode:
    def execute(self, thresh_image, original_image=None, min_area=15000, max_aspect_ratio=2.5, debug_dir=None, debug_prefix=""):
        blocks, error_msg = find_blocks(
            thresh_image, 
            min_area=min_area, 
            max_aspect_ratio=max_aspect_ratio, 
            debug_dir=debug_dir, 
            debug_prefix=debug_prefix, 
            original_img=original_image
        )
        shape = original_image.shape if original_image is not None else thresh_image.shape
        return {"blocks": blocks, "image_shape": shape, "error": error_msg}

from image_processing.preprocessing.onnx_restorer import restore_image_onnx

class ONNXRestorationNode:
    def execute(self, image, model_path="assets/models/RealESR-AnimeVideo-v3_x4.onnx", max_width_before_inference=800, ai_device='CPU'):
        output_bgr, error = restore_image_onnx(image, model_path, max_width_before_inference, ai_device)
        if error:
            return {"image": output_bgr, "error": error}
        return {"image": output_bgr}
