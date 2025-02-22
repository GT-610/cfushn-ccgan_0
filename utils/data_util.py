# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 13:18
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
from tabulate import tabulate


# 查看类别标签与类别索引
def show_class_labels_map(unique_labels: []):
    index_header = ["index"] + [str(i) for i in range(len(unique_labels))]
    data = ["label"] + [str(i) for i in unique_labels]
    print("类标签与索引对应关系:\n", tabulate([data], headers=index_header, tablefmt="pretty"))
    reverse_label_mapping = {original_label: index for index, original_label in
                             enumerate(unique_labels)}
    return reverse_label_mapping


# 查看数据在各标签分布情况
def get_distribution_table(num_log, unique_row_labels, unique_col_labels, cols_per_page=20):
    """
    num_log: 一维列表/数组,长度 = len(unique_row_labels)*len(unique_col_labels).
      假设行优先: 第 r 行, 第 c 列的值在 num_log[r*total_cols + c].
    unique_row_labels: 行标签(通常是 class labels).
    unique_col_labels: 列标签(通常是 cont labels).
    cols_per_page: 每次打印多少列,超出部分拆分为多页.

    返回: 拼接好的字符串,可直接print().
    """
    table_print = ""
    total_cols = len(unique_col_labels)

    start_col = 0

    while start_col < total_cols:
        # 当前分页的列区间
        end_col = min(start_col + cols_per_page, total_cols)
        # 构造表头
        current_headers = ["class\\cont"] + [str(c) for c in unique_col_labels[start_col:end_col]]
        # 构造分页后的表格数据
        table_data = []
        for r, row_label in enumerate(unique_row_labels):
            # 针对本行,把 [start_col, end_col) 这段列的数据取出
            row_slice_start = r * total_cols + start_col
            row_slice_end = r * total_cols + end_col
            row_vals = num_log[row_slice_start:row_slice_end]  # 这是(列数量)大小的一维切片
            # 若 row_vals 不是 Python list, 可以 row_vals.tolist()
            if hasattr(row_vals, 'tolist'):
                row_vals = row_vals.tolist()
            # 该行前面放 row_label, 后面拼上分列数据
            row_data = [row_label] + row_vals
            table_data.append(row_data)

        # --- 使用 tabulate 打印此页的表 ---
        sub_table_str = tabulate(table_data, headers=current_headers, tablefmt="pretty")

        # 在结果中加上页标题(可选),再拼接表格
        table_print += sub_table_str + "\n"

        # 移动到下一段列
        start_col = end_col

    return table_print


if __name__ == "__main__":
    # num_log = [1,3,4,5,6,7]  # ok
    num_log = list(range(6))  # 注意: range返回的是range对象,不是list (range的切片也仍是range对象)
    print(type(range(4)))
    print(get_distribution_table(num_log, range(2), range(3)))

    import torch

    num_log = (100 * torch.randn(150)).numpy().astype(int)
    print(get_distribution_table(num_log, range(1, 4), range(1, 51)))
