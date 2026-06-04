# 凝血定量分析工具

复现 ImageJ 标准工作流：**8-bit → Invert → Rectangle ROI → Measure Mean**

## 两种方式

### 方式一：ImageJ 宏（直接在 ImageJ/Fiji 里用）

1. 用 ImageJ/Fiji 打开图片
2. Plugins > Macros > Run... > 选择 `imagej_workflow.ijm`
3. 在玻片上画矩形框 → 点 OK
4. 结果自动保存到图片旁边的 `results_*/` 文件夹

### 方式二：Python 脚本（自动化批处理）

```bash
# 安装
pip3 install opencv-python numpy

# 单张（自动检测玻片 + 全片分析）
python3 coagulation_analysis.py image.jpg

# 12格分析（3行×4列）
python3 coagulation_analysis.py image.jpg --rows 3 --cols 4

# 手动框选玻片
python3 coagulation_analysis.py image.jpg --manual

# 批量处理
python3 coagulation_analysis.py /path/to/folder/ --batch --rows 3 --cols 4
```

## 工作流对照

| ImageJ 操作 | Python 等价 |
|-------------|------------|
| Image > Type > 8-bit | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| Edit > Invert | `cv2.bitwise_not(gray)` = 255 - gray |
| Rectangle ROI | 自动纹理检测 或 `--manual` 手动 |
| Analyze > Measure | Mean, Area, StdDev, Min, Max, IntDen |

## 输出

| 文件 | 内容 |
|------|------|
| `*_detection.png` | 玻片检测结果（绿框） |
| `*_grid_overlay.png` | 网格划分（黄框=纹理区，绿线=格子） |
| `*_grid_heatmap.png` | 12格反色热力图 |
| `*_cell_XX.png` | 每格原图/灰度/反色对比 |
| `*_results.csv` | 数据表（Excel 直接打开） |
| `*_results.json` | 完整数据 |

## 指标（反色后，值越高 = 凝血越多）

| 指标 | ImageJ 名称 | 含义 |
|------|------------|------|
| Mean | Mean | 平均灰度，核心指标 |
| StdDev | StdDev | 标准差，凝血均匀度 |
| IntDen | IntDen | 积分光密度 = Mean × 像素数 |
| Area | Area | ROI 面积（像素） |
