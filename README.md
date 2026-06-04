# 凝血定量分析工具

## 三个脚本，对应三种场景

| 脚本 | 场景 | 用法 |
|------|------|------|
| `full_workflow.py` | **整张玻片照片**：交互框选 → 自动裁切分格 → 热力图 → 多组对比 | `python3 full_workflow.py 玻片照.jpg` |
| `analyze.py` | **已裁好的方格截图**：拖进去直接分析 | `python3 analyze.py 文件夹/ --watch` |
| `coagulation_analysis.py` | 全自动（旧版，低对比度图可能不准） | `python3 coagulation_analysis.py 图.jpg` |

## full_workflow.py — 推荐主用

```bash
# 交互裁切单张玻片
python3 full_workflow.py 玻片照.jpg --rows 3 --cols 6

# 多组对比（实验组 vs 对照组）
python3 full_workflow.py 图片文件夹/ --compare --rows 3 --cols 6
```

### 操作流程
1. 拖矩形框住所有方格 → 空格确认
2. 预览网格 → 任意键确认，ESC 重来
3. 自动裁切 → ImageJ 精度分析 → 出热力图 + CSV

### 输出
- `*_heatmap.png` — 热力图（蓝=低凝血，红=高凝血）
- `*_results.csv` — 每格 Mean/Median/Std/IntDen
- `cell_*.png` — 裁好的每格图片
- `comparison_results.csv` — 多组对比汇总表（--compare 模式）

## analyze.py — 已有裁切好的方格

```bash
# 监控模式
python3 analyze.py 待分析图片/ --watch

# 批量
python3 analyze.py 待分析图片/ --batch
```

## 指标说明（反色后，值越高=凝血越多）

| 指标 | 含义 |
|------|------|
| Mean | 该格平均灰度，核心指标 |
| Std | 标准差，反映凝血均匀度 |
| IntDen | 积分光密度 = Mean × 像素数 |

## 精度对齐

灰度转换使用 ImageJ 完全一致的公式：`gray = 0.299×R + 0.587×G + 0.114×B`
反色：`255 - gray`
