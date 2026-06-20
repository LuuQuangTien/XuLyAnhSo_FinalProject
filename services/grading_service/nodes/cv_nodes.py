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

import numpy as np

_ONNX_SESSIONS = {}

class ONNXRestorationNode:
    def execute(self, image, model_path="assets/models/RealESR-AnimeVideo-v3_x4.onnx", max_width_before_inference=800, ai_device='CPU'):
        if image is None: return {"image": image}
        
        # Bóp nhỏ ảnh trước khi nội suy x4 để tránh tràn RAM
        h, w = image.shape[:2]
        if w > max_width_before_inference:
            scale = max_width_before_inference / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        global _ONNX_SESSIONS
        cache_key = f"{model_path}_{ai_device}"
        if cache_key not in _ONNX_SESSIONS:
            if not os.path.exists(model_path):
                # Nếu không tìm thấy ONNX, dùng giải pháp sharpen truyền thống
                blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
                sharpened = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)
                return {"image": sharpened}
            try:
                import onnxruntime as ort
                if 'GPU' in ai_device:
                    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                _ONNX_SESSIONS[cache_key] = ort.InferenceSession(model_path, providers=providers)
            except Exception as e:
                return {"image": image, "error": f"ONNX Load Error ({ai_device}): {e}"}

        # Chuẩn bị Tensor (1, 3, H, W) RGB Float32 [0..1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_np = np.array(img_rgb, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)
        
        try:
            session = _ONNX_SESSIONS[cache_key]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_np})
            output = np.squeeze(outputs[0], axis=0)
            
            output = np.clip(output, 0.0, 1.0)
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255.0).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # QUAN TRỌNG: AI Model x4 làm ảnh to ra gấp 4 lần (ví dụ: 800 -> 3200px).
            # Ở kích thước khổng lồ này, thuật toán AdaptiveThreshold (block_size=31) 
            # sẽ bị nhỏ hơn kích thước ô tròn đen, làm các ô tròn bị rỗng/thủng lỗ.
            # Do đó, ta phải bóp ảnh sắc nét này về lại kích thước chuẩn (1500px) để thuật toán CV chạy đúng.
            out_h, out_w = output_bgr.shape[:2]
            if out_w > 1500:
                out_scale = 1500 / out_w
                output_bgr = cv2.resize(output_bgr, (int(out_w * out_scale), int(out_h * out_scale)), interpolation=cv2.INTER_AREA)
            
            return {"image": output_bgr}
        except Exception as e:
            return {"image": image, "error": f"ONNX Inference Error: {e}"}
