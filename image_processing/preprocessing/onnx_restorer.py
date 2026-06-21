import os
import cv2
import numpy as np

_ONNX_SESSIONS = {}

def restore_image_onnx(image, model_path="assets/models/RealESR-AnimeVideo-v3_x4.onnx", max_width_before_inference=800, ai_device='CPU'):
    if image is None: return image, ""
    
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
            return sharpened, ""
        try:
            import onnxruntime as ort
            if 'GPU' in ai_device:
                providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            _ONNX_SESSIONS[cache_key] = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            return image, f"ONNX Load Error ({ai_device}): {e}"

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
        
        return output_bgr, ""
    except Exception as e:
        return image, f"ONNX Inference Error: {e}"
