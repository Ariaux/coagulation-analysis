# 凝血定量分析工具

AI辅助ImageJ工作流的凝血检测定量分析脚本。替代手动ImageJ操作：自动框选玻片 → 划分方格 → 反色 → 测灰度 → 输出数据。

## 安装依赖

```bash
pip3 install opencv-python numpy
```

## 使用方法

### 单张图片（默认3行×4列网格）

```bash
python3 coagulation_analysis.py image.jpg
```

### 指定网格行列

```bash
python3 coagulation_analysis.py image.jpg --rows 3 --cols 4   # 3行×4列=12格
python3 coagulation_analysis.py image.jpg --rows 4 --cols 3   # 4行×3列=12格
```

### 手动框选玻片（如果自动检测不准）

```bash
python3 coagulation_analysis.py image.jpg --manual
```

操作：鼠标依次点击玻片四个角（左上→右上→右下→左下），按空格确认。

### 调整玻片检测大小

```bash
python3 coagulation_analysis.py image.jpg --size 0.25   # 玻片约占画面25%
python3 coagulation_analysis.py image.jpg --size 0.35   # 玻片约占画面35%
```

### 批量处理整个文件夹

```bash
python3 coagulation_analysis.py /path/to/folder/ --batch --rows 3 --cols 4
```

## 输出文件

每个输入图片生成一个 `*_analysis/` 文件夹：

| 文件 | 说明 |
|------|------|
| `*_grid_overlay.png` | 检测结果：黄框=纹理区域，绿线=网格划分 |
| `*_grid_heatmap.png` | 12格反色热力图总览 |
| `*_cell_01.png ~ 12` | 每格单独对比图（原图/灰度/反色） |
| `*_results.csv` | Excel表格：每格的均值、中位数、标准差等 |
| `*_results.json` | 完整数据（含直方图等） |

## 指标说明

所有指标基于**反色后**的灰度值（255 - 原始灰度），反色后**值越高 = 凝血越多**。

| 指标 | 含义 |
|------|------|
| Mean | 该格平均灰度值，核心指标 |
| Median | 中位数，与Mean差距大说明凝血不均匀 |
| Std | 标准差，反映凝血分布均匀程度 |
| IntDen | 积分光密度 = Mean × 像素数 |
| Mean_norm | 归一化均值 (0~1) |

## 工作流程

1. 加载图片
2. 自动检测正方形玻片（纹理梯度法，裁掉空白边距）
3. 在玻片内检测实际纹理区域（黄色框）
4. 纹理区域划分网格（绿色线）
5. 每格独立反色、测灰度
6. 输出统计数据 + 可视化图
