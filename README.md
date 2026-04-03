核心流程：
1. 从数据集表格读取 `id`，筛选已标注的全景街景图。
2. 将全景图转为四向视角并拼接保存。
3. 调用 OpenAI API 对拼接图进行主观感知评分（CQ/AQ/HQ/VQ）。
4. 训练多种监督学习模型并输出对比结果与可视化。

## 使用步骤

### 1) 安装依赖
```bash
pip install -r /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/requirements.txt
```

### 2) 设置 API Key
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

### 3) 筛选已标注街景
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/prepare_labeled_images.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml
```
输出：`../outputs/labeled_panos/`

### 4) 生成四向拼接街景
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/pano_to_stitched.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml
```
输出：`../outputs/stitched_views/`

### 5) MLLM 感知评分（4 组）
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/run_mllm_groups.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml
```

### 6) 监督学习训练与评估
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/train_supervised.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml
```
输出：`../outputs/supervised/`

### 7) 监督学习训练样本规模对比
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/train_supervised_sizes.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml
```
可选参数：
- `--sizes`：训练样本量点位，默认 `10,30,50,70,100,150,200,250,300`
- `--plot-metric`：输出指标曲线，`both`（默认，同时输出 r2+rmse），或 `r2` / `rmse`

输出：
- `../outputs/supervised/metrics_by_train_size.csv`
- `../outputs/supervised/plots/r2_vs_train_size_{CQ|AQ|HQ|VQ}.png`
- `../outputs/supervised/plots/r2_vs_train_size_{CQ|AQ|HQ|VQ}_legend.png`
- 若使用 `--plot-metric rmse`，则文件名为 `rmse_vs_train_size_{...}.png` 与 `_legend.png`

### 8) 对比与可视化
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/compare_and_plot.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml
```
输出：`../outputs/supervised/plots/`

### 9) 分布对比图（Human vs MLLM）
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/plot_distribution_compare.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml \
  --mllm-group baseline
```
输出：`../outputs/supervised/plots/diagnostics/`

### 10) 误差-特征散点图
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/plot_error_feature_scatter.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml \
  --mllm-group baseline
```
可选：指定特征列表  
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/plot_error_feature_scatter.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml \
  --mllm-group baseline \
  --features TRS,TDS,TWS
```
输出：`../outputs/supervised/plots/diagnostics/`

### 11) MLLM 全量推广预测（基线模型）
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/predict_mllm_all_svi.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml \
  --output /Volumes/t7/python_file/LLM_SVIs_exp/outputs/mllm_baseline_full.xlsx \
  --sleep 0.2 \
  --resume
```
输出：`../outputs/mllm_baseline_full.xlsx`（字段结构与数据集一致）

### 12) SHAP 解释（特征贡献）
```bash
pip install shap
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/shap_rf_analysis.py \
  --config /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/configs/default.yaml \
  --sample 300
```
输出：`../outputs/supervised/shap/`

### 13) ALE 特征曲线（叠加对比）
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/ale_mllm_features.py
```
快速试运行（CQ）：
```bash
python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/ale_mllm_features.py --targets CQ
```
输出：  
`../results/ale_curves/overlay/ale_overlay_CQ.png`  
`../results/ale_curves/overlay/ale_overlay_CQ_legend.png`
