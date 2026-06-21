import os
import cv2
import json
import tempfile
import numpy as np

def get_ai_paper_bounding_box(image, log_txt, debug_prefix=""):
    """
    Sử dụng model AI (silueta.onnx) để tìm bounding box của tờ giấy.
    """
    model_path = os.path.join(os.getcwd(), "assets", "models", "silueta.onnx")
    if not os.path.exists(model_path):
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[AI SILUETA] Không tìm thấy silueta.onnx!\n")
        return None

    try:
        bx, by, bw, bh = None, None, None, None
        cache_file = ""
        
        # KIỂM TRA CACHE TRƯỚC KHI CHẠY AI (Lưu ý: Thêm kích thước ảnh vào key để tránh lỗi khi upscale)
        if debug_prefix:
            height, width = image.shape[:2]
            cache_file = os.path.join(tempfile.gettempdir(), f"omr_ai_cache_{debug_prefix}_{width}x{height}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as fc:
                        data = json.load(fc)
                    bx, by, bw, bh = data['bx'], data['by'], data['bw'], data['bh']
                    with open(log_txt, "a", encoding="utf-8") as f:
                        f.write(f"[AI SILUETA] Đã sử dụng Bounding Box từ CACHE: {bx, by, bw, bh}\n")
                except: pass
                
        # NẾU KHÔNG CÓ TRONG CACHE -> CHẠY AI
        if bx is None:
            import onnxruntime as ort
            height, width = image.shape[:2]
            
            # Cấu hình để tắt các cảnh báo không cần thiết từ onnxruntime
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            
            # Tải model vào onnxruntime
            net = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            input_name = net.get_inputs()[0].name
            
            # Đưa ảnh vào mạng AI (Sử dụng OpenCV blobFromImage để tiền xử lý dễ dàng)
            # OpenCV blob = (image - mean) * scalefactor. Mean ở đây là [123.675, 116.28, 103.53] (tương đương 0.485, 0.456, 0.406)
            blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (320, 320), (0.485*255, 0.456*255, 0.406*255), swapRB=True, crop=False)
            
            # Chia cho std (0.229, 0.224, 0.225) chuẩn của ImageNet
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            blob = blob / std
            
            # Chạy Inference bằng ONNX Runtime
            out = net.run(None, {input_name: blob})[0]
            
            mask = out[0, 0, :, :]
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            
            # Lấy Bounding Box của vật thể lớn nhất
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                
                # LƯU VÀO CACHE CHO LẦN SAU
                if cache_file:
                    try:
                        with open(cache_file, "w") as fc:
                            json.dump({'bx': bx, 'by': by, 'bw': bw, 'bh': bh}, fc)
                    except: pass
        
        if bx is not None:
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[AI SILUETA] Bounding Box: x={bx}, y={by}, w={bw}, h={bh}\n")
            return (bx, by, bw, bh)
            
    except Exception as e:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[AI SILUETA] Lỗi khi chạy AI Silueta: {e}.\n")
            
    return None
