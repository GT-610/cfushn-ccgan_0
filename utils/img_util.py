import numpy as np
import torch
from torchvision.utils import make_grid


def img_with_sep(imgs_data, nrow, h_sep_gap=1, v_sep_gap=0, width=2):
    """
    生成一个带有分隔线的图像网格：
      - 将输入的 imgs_data (N, C, H, W) 按 nrow 张图像一行排列；
      - 每隔 h_sep_gap 行插入一条水平分隔线（白色，分隔线高度为 width 像素）；
      - 每隔 v_sep_gap 列插入一条垂直分隔线（白色，分隔线宽度为 width 像素），
        若 v_sep_gap=0，则不插入垂直分隔线。

    Args:
        imgs_data (Tensor): 生成图像张量，形状 (N, C, H, W)。
        nrow (int): 每行的图像数量。
        h_sep_gap (int, optional): 每隔多少行插入一条水平分隔线，默认 5 行。
        v_sep_gap (int, optional): 每隔多少列插入一条垂直分隔线，默认 0（不插）。
        width (int, optional): 分隔线的像素宽度，默认 1。

    Returns:
        Tensor: 带有分隔线的图像网格，形状 (C, H_total, W_total)。
    """
    # 先使用 make_grid 构建基本网格，不使用 padding
    grid = make_grid(imgs_data, nrow=nrow, padding=0)  # grid: (C, H_total, W_total)
    C, H_total, W_total = grid.shape

    # 计算网格行数：N = imgs_data.shape[0]，行数 = ceil(N / nrow)
    N = imgs_data.shape[0]
    num_rows = int(np.ceil(N / nrow))
    # 假设每张图像尺寸相同，则每行图像高度 H = H_total // num_rows
    H = H_total // num_rows
    # 每列图像宽度 W = W_total // nrow
    W = W_total // nrow

    # 将 grid 按行分割成列表
    rows = []
    for i in range(num_rows):
        row_img = grid[:, i * H:(i + 1) * H, :]
        # 若需要插入垂直分隔线，则对当前行进行处理
        if v_sep_gap and v_sep_gap > 0:
            cols = []
            for j in range(nrow):
                col_img = row_img[:, :, j * W:(j + 1) * W]
                cols.append(col_img)
                # 如果当前列之后需要插入垂直分隔线
                if (j + 1) % v_sep_gap == 0 and (j + 1) < nrow:
                    sep_v = torch.ones((C, H, width), device=grid.device)  # 白色分隔线
                    cols.append(sep_v)
            # 拼接当前行各列，方向为宽度方向
            row_img = torch.cat(cols, dim=2)
        rows.append(row_img)
        # 每隔 h_sep_gap 行（且不是最后一行），插入一条水平分隔线
        if (i + 1) % h_sep_gap == 0 and (i + 1) < num_rows:
            # 生成水平分隔线，大小 (C, width, 当前行的宽度)
            sep_h = torch.ones((C, width, row_img.shape[2]), device=grid.device)
            rows.append(sep_h)

    # 将所有行拼接，方向为高度方向
    grid_with_sep = torch.cat(rows, dim=1)
    return grid_with_sep


# 示例调用：
if __name__ == "__main__":
    # 随机生成 250 张图像，假设图像尺寸为 3x64x64
    imgs = torch.randn(250, 3, 64, 64)
    # 设定每行显示 10 张图像，分隔每 5 行添加一条水平分隔线
    grid_img = img_with_sep(imgs, nrow=10, h_sep_gap=5, v_sep_gap=1, width=2)
    # 使用 torchvision 的 save_image 保存结果
    from torchvision.utils import save_image

    save_image(grid_img, "grid_with_sep.png", normalize=True)