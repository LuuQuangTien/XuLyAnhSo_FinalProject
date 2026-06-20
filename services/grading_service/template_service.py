import os
import json

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'templates')

class TemplateService:
    @staticmethod
    def get_all_templates():
        """Đọc tất cả các file .json cấu hình Pipeline từ thư mục templates/"""
        if not os.path.exists(TEMPLATE_DIR):
            os.makedirs(TEMPLATE_DIR)
            
        templates = []
        for file in os.listdir(TEMPLATE_DIR):
            if file.endswith('.json'):
                path = os.path.join(TEMPLATE_DIR, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Basic Schema Validation (DAG Engine)
                        if "nodes" not in data:
                            print(f"[TemplateService] Bỏ qua {file}: Thiếu thuộc tính 'nodes' cho DAG Engine.")
                            continue
                            
                        data['_filepath'] = path
                        templates.append(data)
                except Exception as e:
                    print(f"[TemplateService] Lỗi khi nạp {file}: {e}")
        return templates
