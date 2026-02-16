# Pipeline 逻辑梳理

## 两种运行模式

### 1. 小模型（合成模拟数据）— 与文章图一致

- **数据来源**：无 `data/synthetic_sample_data.xlsx` 时，自动生成 `synthetic_sample_data.csv`（合成数据）。
- **Pipeline**：用合成数据跑轨迹 GMM、特征、二分类模型、生存分析，得到 `figure2_*`、`figure3_*`、`figure4_km.png` 等。
- **Main / Supplementary 图**：**用模拟数据** 生成（`run_article_figures.py`），曲线和数值为**参数化/硬编码**，与文章中的图**完全一致**。
- **用途**：复现文章、审稿/投稿用图。

### 2. 完整大模型（真实/原始数据）— 图来自真实结果

- **数据来源**：存在 `data/synthetic_sample_data.xlsx` 时，**优先使用该原始数据**（客户/真实数据）。
- **Pipeline**：用**真实数据**跑轨迹、特征、模型、生存分析，得到的是**真实结果**对应的 `figure2_*`、`figure3_*`、`figure4_km.png` 等。
- **Main / Supplementary 图**：必须用 **pipeline 的真实结果** 生成（`figures_from_pipeline.py`），即：
  - Main_Figure1/2、Supplementary_Figure1–8 的数据都来自本次运行的 `df_traj`、`model_out`、`surv_out`、Cox 等；
  - **不再**使用 `run_article_figures.py` 的模拟数据。
- **用途**：真实数据分析、客户报告；图与文章不一致是预期行为。

## 当前实现（代码分支）

```
sepsis_trajectory.py main():
  if xlsx 存在:
    使用 xlsx 真实数据
    → 跑完整 pipeline（轨迹/特征/模型/生存）
    → 调用 figures_from_pipeline.gen_all(results_dir, df_traj, traj_out, model_out, surv_out, ...)
       → Main_Figure1/2、Supplementary_Figure1–8 全部由 pipeline 结果绘制
  else:
    生成或使用 synthetic_sample_data.csv
    → 跑 pipeline
    → 调用 run_article_figures.main(results_dir)
       → Main/Supp 图用模拟数据，与文章一致
```

## 小结

| 项目           | 小模型（合成数据）     | 完整大模型（真实数据）   |
|----------------|------------------------|--------------------------|
| 数据           | 合成 CSV               | 原始 xlsx                |
| Pipeline 图    | 来自合成数据           | 来自真实数据             |
| Main/Supp 图   | 模拟数据（与文章一致） | **Pipeline 真实结果**   |
| efigure 数据源 | 模拟                   | **真实数据 + 完整模型**  |

## GitHub 上的 results/

- 仓库中提交的 `results/` 建议为 **模拟数据跑出的全部图**（与文章一致），便于复现与审稿。
- 操作：在 **无** `data/synthetic_sample_data.xlsx` 时运行 `python sepsis_trajectory.py`，生成 Main_Figure1/2 与 Supplementary_Figure1–8 后，将 `results/` 提交并推送到 GitHub。
