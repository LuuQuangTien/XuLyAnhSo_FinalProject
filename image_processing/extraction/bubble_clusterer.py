"""
Bubble Clusterer — Thuật toán Gom cụm thông minh (Structural Clustering).

Module này nhận danh sách bong bóng thô (x, y, w, h) và tự động phân loại
chúng thành các khu vực có ý nghĩa dựa trên CẤU TRÚC HÌNH HỌC, không hardcode
tọa độ tuyệt đối:
  - SBD (Số Báo Danh): Grid ~10 hàng, nhiều cột (>=4)
  - Mã Đề: Grid ~10 hàng, ít cột (2-3)
  - MCQ (Đáp án trắc nghiệm): Grid nhiều hàng (>=15), 4 cột (A, B, C, D)

Quy tắc phân biệt:
  - Không dùng tọa độ tuyệt đối → hoạt động cho mọi layout phiếu thi.
  - Dựa vào số hàng/cột của mỗi grid để tự nhận diện loại.
"""
import numpy as np


def cluster_bubbles(raw_bubbles):
    """
    Entry point chính. Nhận danh sách bong bóng thô, trả về dict blocks có cấu trúc.
    
    Args:
        raw_bubbles: list[dict] — mỗi dict có keys: x, y, w, h
    Returns:
        dict — blocks đã được gán nhãn (SBD_Block, Ma_De_Block, Answers_Col_N...)
    """
    if not raw_bubbles or len(raw_bubbles) < 10:
        return _fallback_block(raw_bubbles)

    # Bước 1: Loại bỏ bong bóng trùng lặp (giữ ô lớn hơn)
    clean = _deduplicate(raw_bubbles)

    # Bước 2: Gom thành các hàng ngang theo tọa độ Y
    rows = _cluster_rows(clean)

    # Bước 3: Tách mỗi hàng thành các đoạn (segments) theo khoảng trống X
    grids = _detect_grids(rows)

    # Bước 4: Phân tích cấu trúc và gán nhãn cho mỗi grid
    blocks = _label_grids(grids)

    return blocks if blocks else _fallback_block(raw_bubbles)


# ============================================================
# BƯỚC 1: Loại bỏ trùng lặp
# ============================================================
def _deduplicate(bubbles, dist_thresh=5):
    """Nếu 2 ô có tâm cách nhau < dist_thresh px, giữ lại ô LỚN hơn."""
    enriched = []
    for b in bubbles:
        cx = b['x'] + b['w'] / 2
        cy = b['y'] + b['h'] / 2
        enriched.append({**b, 'cx': cx, 'cy': cy, 'area': b['w'] * b['h']})

    enriched.sort(key=lambda b: b['area'], reverse=True)

    kept = []
    for bubble in enriched:
        is_dup = any(
            abs(bubble['cx'] - e['cx']) < dist_thresh and abs(bubble['cy'] - e['cy']) < dist_thresh
            for e in kept
        )
        if not is_dup:
            kept.append(bubble)
    return kept


# ============================================================
# BƯỚC 2: Gom hàng ngang (Y-Clustering)
# ============================================================
def _cluster_rows(bubbles):
    """Gom các bong bóng thành hàng ngang dựa trên tọa độ Y."""
    if not bubbles:
        return []

    heights = [b['h'] for b in bubbles]
    tolerance = np.median(heights) * 0.6

    sorted_b = sorted(bubbles, key=lambda b: b['cy'])
    rows = []
    current_row = [sorted_b[0]]

    for b in sorted_b[1:]:
        row_median_y = np.median([rb['cy'] for rb in current_row])
        if abs(b['cy'] - row_median_y) <= tolerance:
            current_row.append(b)
        else:
            rows.append(sorted(current_row, key=lambda b: b['cx']))
            current_row = [b]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b['cx']))
    return rows


# ============================================================
# BƯỚC 3: Tách hàng → Grid bằng X-gap
# ============================================================
def _detect_grids(rows):
    """
    Phát hiện các grid riêng biệt bằng cách tìm khoảng trống lớn 
    trong mỗi hàng (X-gap), rồi gom các đoạn thẳng hàng dọc lại.
    """
    if not rows:
        return []

    # Tính khoảng cách trung vị giữa các bong bóng liền kề trong hàng
    all_spacings = []
    for row in rows:
        for i in range(len(row) - 1):
            sp = row[i + 1]['cx'] - row[i]['cx']
            if sp > 0:
                all_spacings.append(sp)

    if not all_spacings:
        return [rows]

    median_spacing = np.median(all_spacings)
    gap_threshold = median_spacing * 2.0  # Khoảng trống > 2x median = ranh giới grid

    # Tách mỗi hàng thành segments tại các gap lớn
    all_segments = []
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        cur_seg = [row[0]]
        for i in range(1, len(row)):
            if (row[i]['cx'] - row[i - 1]['cx']) > gap_threshold:
                all_segments.append((row_idx, cur_seg))
                cur_seg = [row[i]]
            else:
                cur_seg.append(row[i])
        if cur_seg:
            all_segments.append((row_idx, cur_seg))

    # Gom các segment thẳng hàng dọc (X-range tương tự) thành grid
    grid_groups = []  # Mỗi group là list các segment index

    for seg_idx, (row_idx, seg) in enumerate(all_segments):
        seg_xmin = min(b['cx'] for b in seg)
        seg_xmax = max(b['cx'] for b in seg)

        matched = None
        for g_idx, group in enumerate(grid_groups):
            # Lấy X-range trung bình của group
            g_xmins = []
            g_xmaxs = []
            for other_idx in group:
                _, other_seg = all_segments[other_idx]
                g_xmins.append(min(b['cx'] for b in other_seg))
                g_xmaxs.append(max(b['cx'] for b in other_seg))

            g_xmin_avg = np.mean(g_xmins)
            g_xmax_avg = np.mean(g_xmaxs)

            # Kiểm tra overlap: segment mới có nằm trong vùng X của group không
            overlap_min = max(seg_xmin, g_xmin_avg)
            overlap_max = min(seg_xmax, g_xmax_avg)
            seg_range = seg_xmax - seg_xmin

            if seg_range > 0 and (overlap_max - overlap_min) / seg_range > 0.5:
                matched = g_idx
                break

        if matched is not None:
            grid_groups[matched].append(seg_idx)
        else:
            grid_groups.append([seg_idx])

    # Chuyển đổi grid_groups → cấu trúc rows cho mỗi grid
    grids = []
    for group in grid_groups:
        grid_rows_dict = {}
        for seg_idx in group:
            row_idx, seg = all_segments[seg_idx]
            if row_idx not in grid_rows_dict:
                grid_rows_dict[row_idx] = []
            grid_rows_dict[row_idx].extend(seg)

        sorted_grid = []
        for ri in sorted(grid_rows_dict.keys()):
            sorted_grid.append(sorted(grid_rows_dict[ri], key=lambda b: b['cx']))
        grids.append(sorted_grid)

    return grids


# ============================================================
# BƯỚC 4: Gán nhãn dựa trên cấu trúc (rows × cols)
# ============================================================
def _label_grids(grids):
    """
    Phân tích mỗi grid và gán nhãn:
      - 10 hàng → Numeric (SBD hoặc Mã Đề)
      - Nhiều hàng, 4 cột → MCQ
      - Grid quá nhỏ → bỏ qua (nhiễu / anchor)
    """
    blocks = {}
    mcq_grids = []
    numeric_grids = []

    for grid in grids:
        num_rows = len(grid)
        col_counts = [len(row) for row in grid]
        if not col_counts:
            continue
        num_cols = int(np.median(col_counts))
        total = sum(len(row) for row in grid)

        # Bỏ qua grid quá nhỏ (nhiễu, anchor lẻ)
        if total < 8:
            continue

        if 8 <= num_rows <= 12 and num_cols <= 10:
            # Grid ~10 hàng → Vùng điền số (SBD hoặc Mã Đề)
            numeric_grids.append({'grid': grid, 'rows': num_rows, 'cols': num_cols})
        else:
            # Mặc định: coi là vùng trắc nghiệm
            mcq_grids.append({'grid': grid, 'rows': num_rows, 'cols': num_cols})

    # === Gán nhãn Numeric grids ===
    # Grid có nhiều cột hơn = SBD, ít cột hơn = Mã Đề
    numeric_grids.sort(key=lambda g: g['cols'], reverse=True)
    for i, ng in enumerate(numeric_grids):
        if i == 0 and ng['cols'] >= 4:
            name, btype = "SBD_Block", "sbd"
        else:
            name, btype = "Ma_De_Block", "made"

        blocks[name] = {
            "type": btype,
            "rows": ng['rows'],
            "cols": ng['cols'],
            "bubbles": _flatten_grid(ng['grid'])
        }

    # === Gán nhãn MCQ grids ===
    # Sắp xếp từ trái sang phải (theo min X) để đánh số câu hỏi
    mcq_grids.sort(key=lambda g: min(b['cx'] for row in g['grid'] for b in row))
    start_q = 1
    for i, mg in enumerate(mcq_grids):
        blocks[f"Answers_Col_{i + 1}"] = {
            "type": "mcq",
            "start_question": start_q,
            "rows": mg['rows'],
            "cols": mg['cols'],
            "bubbles": _flatten_grid(mg['grid'])
        }
        start_q += mg['rows']

    return blocks


def _flatten_grid(grid):
    """Chuyển grid (list of rows) thành flat list có gắn row/col index."""
    result = []
    for row_idx, row in enumerate(grid):
        for col_idx, b in enumerate(row):
            result.append({
                "x": b['x'], "y": b['y'], "w": b['w'], "h": b['h'],
                "row": row_idx, "col": col_idx
            })
    return result


def _fallback_block(bubbles):
    """Trả về block mặc định khi clustering thất bại."""
    return {
        "Auto_Detected_Block": {
            "type": "mcq_grid",
            "format": "freeform_bubbles",
            "bubbles": bubbles or []
        }
    }
