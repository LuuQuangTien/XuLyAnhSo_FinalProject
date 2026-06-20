import os
import pandas as pd

class AnswerKeyService:
    @staticmethod
    def load_answers(filepath="assets/sample_answers.xlsx"):
        """Nạp đáp án đúng từ file Excel."""
        try:
            if not os.path.exists(filepath):
                return {}
            df = pd.read_excel(filepath).astype(str)
            if df.empty: return {}
            
            answers = {}
            question_col = df.columns[0]
            # Hỗ trợ nhiều mã đề. Ví dụ dict trả về: {'Mã Đề 000': {'1': 'A', '2': 'B'}}
            
            if len(df.columns) < 2: return {}
            
            for col in df.columns[1:]:
                ans_map = dict(zip(df[question_col], df[col]))
                # Lấy mã đề bằng cách xóa chữ 'Mã Đề'
                col_str = str(col).replace('Mã Đề', '').strip()
                answers[col_str] = {k.replace('.0', ''): v.upper() for k, v in ans_map.items() if v != 'nan' and v.strip() != ''}
            return answers
        except Exception as e:
            print(f"Lỗi khi đọc file đáp án Excel: {e}")
            return {}

    @staticmethod
    def get_dataframe(filepath: str) -> pd.DataFrame:
        """Đọc file Excel thành pandas DataFrame dùng cho UI."""
        if not os.path.exists(filepath):
            return pd.DataFrame()
        return pd.read_excel(filepath).astype(str)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str):
        """Lưu pandas DataFrame xuống file Excel."""
        df.to_excel(filepath, index=False)
