import pandas as pd

# Mock data
data = {
    "Câu hỏi": [str(i) for i in range(1, 29)],
    "Mã Đề 000": ["A" for _ in range(28)]
}
df = pd.DataFrame(data)
df.to_excel("test_excel.xlsx", index=False)

try:
    df_read = pd.read_excel("test_excel.xlsx").astype(str)
    if df_read.empty:
        print("Empty DataFrame")
    else:
        question_col = df_read.columns[0]
        if len(df_read.columns) == 2:
            ans_map = dict(zip(df_read[question_col], df_read[df_read.columns[1]]))
            res = {k.replace('.0', ''): v.upper() for k, v in ans_map.items() if v != 'nan' and v.strip() != ''}
            print(f"Flat dict: {res}")
        else:
            answers = {}
            for col in df_read.columns[1:]:
                ans_map = dict(zip(df_read[question_col], df_read[col]))
                answers[col] = {k.replace('.0', ''): v.upper() for k, v in ans_map.items() if v != 'nan' and v.strip() != ''}
            print(f"Nested dict: {answers}")
except Exception as e:
    print(f"Error: {e}")
