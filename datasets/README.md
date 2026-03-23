# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: `deception_evals`

### Overview
- **Source**: `notrichardren/deception-evals`
- **Size**: 924 records, 70.9 KB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: train (924)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('notrichardren/deception-evals')
dataset.save_to_disk('datasets/deception_evals')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/deception_evals')
```

### Sample Data
- Saved sample file: `datasets/deception_evals/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 2: `truthful_qa_generation`

### Overview
- **Source**: `truthfulqa/truthful_qa` (config: `generation`)
- **Size**: 817 records, 474.5 KB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: validation (817)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('truthfulqa/truthful_qa', 'generation')
dataset.save_to_disk('datasets/truthful_qa_generation')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/truthful_qa_generation')
```

### Sample Data
- Saved sample file: `datasets/truthful_qa_generation/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 3: `truthful_qa_multiple_choice`

### Overview
- **Source**: `truthfulqa/truthful_qa` (config: `multiple_choice`)
- **Size**: 817 records, 610.6 KB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: validation (817)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')
dataset.save_to_disk('datasets/truthful_qa_multiple_choice')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/truthful_qa_multiple_choice')
```

### Sample Data
- Saved sample file: `datasets/truthful_qa_multiple_choice/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 4: `halueval_qa`

### Overview
- **Source**: `pminervini/HaluEval` (config: `qa`)
- **Size**: 10000 records, 5.2 MB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: data (10000)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('pminervini/HaluEval', 'qa')
dataset.save_to_disk('datasets/halueval_qa')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/halueval_qa')
```

### Sample Data
- Saved sample file: `datasets/halueval_qa/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 5: `halueval_general`

### Overview
- **Source**: `pminervini/HaluEval` (config: `general`)
- **Size**: 4507 records, 2.8 MB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: data (4507)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('pminervini/HaluEval', 'general')
dataset.save_to_disk('datasets/halueval_general')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/halueval_general')
```

### Sample Data
- Saved sample file: `datasets/halueval_general/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 6: `sycophancy_raw`

### Overview
- **Source**: `EleutherAI/sycophancy` (config: `raw_snapshot`)
- **Size**: N/A records, 7.8 KB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: N/A
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('EleutherAI/sycophancy', 'raw_snapshot')
dataset.save_to_disk('datasets/sycophancy_raw')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/sycophancy_raw')
```

### Sample Data
- No sample file generated

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 7: `small_sycophancy_dataset`

### Overview
- **Source**: `Alamerton/small-sycophancy-dataset`
- **Size**: 44 records, 10.7 KB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: train (44)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('Alamerton/small-sycophancy-dataset')
dataset.save_to_disk('datasets/small_sycophancy_dataset')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/small_sycophancy_dataset')
```

### Sample Data
- Saved sample file: `datasets/small_sycophancy_dataset/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 8: `sycophancy_answer`

### Overview
- **Source**: `dwb2023/sycophancy-answer`
- **Size**: 7268 records, 6.5 MB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: train (7268)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('dwb2023/sycophancy-answer')
dataset.save_to_disk('datasets/sycophancy_answer')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/sycophancy_answer')
```

### Sample Data
- Saved sample file: `datasets/sycophancy_answer/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Dataset 9: `open_ended_sycophancy`

### Overview
- **Source**: `henrypapadatos/Open-ended_sycophancy`
- **Size**: 53 records, 51.9 KB
- **Format**: HuggingFace Dataset saved to disk
- **Task**: Truthfulness/hallucination/deception/sycophancy evaluation
- **Splits**: train (53)
- **License**: Check original dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset('henrypapadatos/Open-ended_sycophancy')
dataset.save_to_disk('datasets/open_ended_sycophancy')
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/open_ended_sycophancy')
```

### Sample Data
- Saved sample file: `datasets/open_ended_sycophancy/samples/samples.json`

### Notes
- Verify schema before training/evaluation.
- Some datasets are synthetic or prompt-derived and should be interpreted cautiously.

## Failed/Blocked Dataset Attempts

- `truthfulqa/truthful_qa` without config failed (requires `generation` or `multiple_choice`).
- `pminervini/HaluEval` without config failed (requires named config).
- `EleutherAI/sycophancy` loader path failed because only legacy dataset script is provided in this environment.
- `meg-tong/sycophancy-eval` download returned generation error.
