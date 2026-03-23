# Detect Different Lies (NLP)

This project evaluates whether LLM false outputs are primarily epistemic failures or incentive-driven misreporting. It runs a paired control-vs-pressure protocol on factual QA items using real GPT-4.1 API calls and reports mechanism-separated outcomes.

## Key Findings
- Pressure prompts increased false outputs from 12.9% to 40.8% (+27.9 pp).
- Among pressure-condition false outputs (`n=98`), 68.4% were incentive-induced and 31.6% were epistemic.
- Both datasets showed significant paired degradation under pressure (McNemar p < 1e-7 each).
- Cross-mechanism detector transfer was weaker than pooled in-domain performance (AUROC ~0.57-0.60 vs 0.66).

## Reproduce
1. Activate environment:
```bash
source .venv/bin/activate
```
2. Install dependencies (if needed):
```bash
uv pip install -r requirements.txt
```
3. Run main experiment:
```bash
python src/run_mechanism_study.py --model gpt-4.1 --n-truthfulqa 120 --n-halueval 120 --seed 42
```

## Outputs
- Full report: `REPORT.md`
- Planning: `planning.md`
- Main metrics: `results/metrics.json`
- Raw model outputs: `results/raw_outputs.jsonl`
- Paired item table: `results/paired_item_results.csv`
- Data quality summary: `results/data_quality.json`
- Plots: `results/plots/`

## File Structure
- `src/run_mechanism_study.py`: end-to-end experiment runner
- `results/`: metrics, plots, intermediate artifacts
- `datasets/`: pre-downloaded datasets
- `code/`: cloned baseline repositories
