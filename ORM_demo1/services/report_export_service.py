import os
import pandas as pd

class ReportExportService:
    @staticmethod
    def export_single_result(score: dict, base_name: str, output_dir: str):
        """Xuất kết quả chấm 1 bài thi ra file Excel."""
        excel_data = [{
            "Tên file": base_name,
            "Số báo danh": score.get('sbd', 'N/A'),
            "Mã đề thi": score.get('made', 'N/A'),
            "Số câu đúng": f"{score.get('correct', 0)}/{score.get('total', 0)}",
            "Điểm": round(score.get('final_score', 0), 2),
            "Ghi chú": score.get('notes', '')
        }]
        df = pd.DataFrame(excel_data)
        excel_path = os.path.join(output_dir, f"ket_qua_{base_name}.xlsx")
        try:
            df.to_excel(excel_path, index=False)
        except Exception as e:
            print(f"Lỗi xuất file Excel: {e}")

    @staticmethod
    def export_batch_results(excel_data: list, excel_path: str):
        """Xuất kết quả tổng hợp nhiều bài thi ra file Excel."""
        if not excel_data:
            return
        df = pd.DataFrame(excel_data)
        try:
            df.to_excel(excel_path, index=False)
        except Exception as e:
            print(f"Không thể xuất file tổng hợp: {e}")
