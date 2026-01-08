# ARES Evaluation Framework - Learning Plan

## Overview

ARES (Automated RAG Evaluation System) is an open-source framework for evaluating Retrieval-Augmented Generation (RAG) models. It automates evaluation through synthetic data generation, classifier training, and Prediction Powered Inference (PPI).

---

## Documentation URLs & Summaries

### Getting Started

| URL | Summary |
|-----|---------|
| https://ares-ai.vercel.app/getting_started.html | Main overview: ARES framework introduction and navigation |
| https://ares-ai.vercel.app/installation.html | Installation: `pip install ares-ai`, GitHub clone, API key setup |
| https://ares-ai.vercel.app/setup.html | Prerequisites and environment configuration |
| https://ares-ai.vercel.app/quick_start_guide_1.html | Quick Start A: GPT-3.5 evaluation with UES/IDP and PPI |
| https://ares-ai.vercel.app/quick_start_guide_2.html | Quick Start B: Full workflow (synth gen, training, PPI) |

### Core Features

| URL | Summary |
|-----|---------|
| https://ares-ai.vercel.app/ues_idp.html | UES/IDP: Unlabeled evaluation with in-domain prompts for context/answer relevance |
| https://ares-ai.vercel.app/local_model_execution.html | Local model execution using vLLM integration |

### Synthetic Data Generation

| URL | Summary |
|-----|---------|
| https://ares-ai.vercel.app/synth_gen.html | Synthetic generation starter: config, model selection, document sampling |
| https://ares-ai.vercel.app/synth_gen_params.html | All synth gen parameters: filepaths, models, temperatures, ratios |

### Classifier Training

| URL | Summary |
|-----|---------|
| https://ares-ai.vercel.app/training_classifier.html | Training classifiers for relevance/faithfulness assessment |
| https://ares-ai.vercel.app/training_classifier_params.html | All training params: epochs, patience, learning rate, batch size |

### RAG Evaluation

| URL | Summary |
|-----|---------|
| https://ares-ai.vercel.app/rag_eval.html | RAG evaluation configuration with PPI |
| https://ares-ai.vercel.app/rag_eval_params.html | All eval params: checkpoints, labels, alpha, trials |

### Resources

| URL | Summary |
|-----|---------|
| https://ares-ai.vercel.app/datasets.html | KILT and SuperGLUE datasets for evaluation |
| https://ares-ai.vercel.app/tutorial_01.html | Full RAG eval tutorial with PPI on NQ dataset |
| https://ares-ai.vercel.app/faq.html | FAQs: what is ARES, how it works, requirements, metrics |

---

## Core Evaluation Metrics

ARES evaluates three primary dimensions:

1. **Context Relevance** - Does the retrieved information pertain to the query?
2. **Answer Faithfulness** - Is the response grounded without hallucinations?
3. **Answer Relevance** - Does the response appropriately address the query?

---

## ARES Workflow (4 Phases)

### Phase 1: Synthetic Data Generation
- Generate synthetic question-answer pairs from documents
- Configure document sampling and filtering (min 50 words)
- Use few-shot prompts to guide generation

### Phase 2: Classifier Training
- Fine-tune lightweight LM (DeBERTa) as evaluation judge
- Train on synthetic data with validation set
- Output: checkpoint files for evaluation

### Phase 3: RAG Evaluation with PPI
- Apply Prediction Powered Inference
- Combine model predictions with human annotations
- Generate confidence intervals for metrics

### Phase 4: UES/IDP (Alternative)
- Use unlabeled data with in-domain prompts
- Quick evaluation without training
- LLM-based scoring (GPT-3.5/4)

---

## Code Structure for Learning

```
llm_evaluation_ares/
├── docs/
│   └── plan.md                    # This file
├── examples/
│   ├── 01_installation/
│   │   └── setup.py               # Environment setup & verification
│   ├── 02_ues_idp/
│   │   ├── basic_ues_idp.py       # Simple UES/IDP evaluation
│   │   └── custom_prompts.py      # Custom in-domain prompts
│   ├── 03_synthetic_generation/
│   │   ├── basic_synth.py         # Basic synthetic data generation
│   │   ├── custom_model.py        # Using different models
│   │   └── advanced_params.py     # Temperature, percentiles, negatives
│   ├── 04_classifier_training/
│   │   ├── train_relevance.py     # Context relevance classifier
│   │   ├── train_faithfulness.py  # Answer faithfulness classifier
│   │   └── multi_label.py         # Multi-label training
│   ├── 05_rag_evaluation/
│   │   ├── basic_ppi.py           # Basic PPI evaluation
│   │   ├── custom_checkpoints.py  # Using custom classifiers
│   │   └── confidence_intervals.py # Understanding CI outputs
│   ├── 06_full_pipeline/
│   │   ├── end_to_end.py          # Complete ARES pipeline
│   │   └── nq_tutorial.py         # NQ dataset walkthrough
│   ├── 07_local_models/
│   │   ├── vllm_setup.py          # vLLM configuration
│   │   └── local_eval.py          # Local model evaluation
│   └── 08_datasets/
│       ├── kilt_datasets.py       # KILT dataset usage
│       └── superglue_datasets.py  # SuperGLUE dataset usage
├── data/
│   ├── few_shot_prompts/          # Example prompts
│   └── sample_documents/          # Sample docs for testing
├── checkpoints/                   # Saved model checkpoints
├── pyproject.toml
└── README.md
```

---

## Implementation Plan

### Phase 1: Setup & Basic UES/IDP
1. Install ARES: `uv add ares-ai`
2. Configure API keys (OpenAI, TogetherAI, Anthropic)
3. Run basic UES/IDP evaluation
4. Understand in-domain prompts structure

### Phase 2: Synthetic Data Generation
1. Prepare document corpus (TSV format)
2. Create few-shot prompt files
3. Configure synth_config parameters
4. Generate synthetic queries and validate output

### Phase 3: Classifier Training
1. Set up training configuration
2. Train context relevance classifier
3. Train answer faithfulness classifier
4. Evaluate classifier performance

### Phase 4: PPI Evaluation
1. Configure PPI with trained checkpoints
2. Run evaluation on unlabeled data
3. Interpret confidence intervals
4. Compare with ground truth

### Phase 5: Local Models
1. Set up vLLM server
2. Configure local model execution
3. Compare local vs API results

### Phase 6: Advanced Topics
1. Multi-label training
2. Custom datasets (KILT, SuperGLUE)
3. Production deployment considerations

---

## Key Code Examples

### UES/IDP Evaluation
```python
from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "unlabeled_evaluation_set": "nq_unlabeled_output.tsv",
    "model_choice": "gpt-3.5-turbo-0125"
}

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
# Returns: context relevance, answer faithfulness, answer relevance scores
```

### Synthetic Data Generation
```python
from ares import ARES

synth_config = {
    "document_filepaths": ["data/documents.tsv"],
    "few_shot_prompt_filename": "data/few_shot_prompts.tsv",
    "synthetic_queries_filenames": ["output/synthetic_queries.tsv"],
    "model_choice": "google/flan-t5-xxl",
    "documents_sampled": 1000
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
```

### Classifier Training
```python
from ares import ARES

classifier_config = {
    "classification_dataset": ["output/synthetic_queries.tsv"],
    "test_set_selection": "data/validation.tsv",
    "label_column": ["Context_Relevance_Label"],
    "model_choice": "microsoft/deberta-v3-large",
    "num_epochs": 10,
    "patience_value": 3,
    "learning_rate": 5e-6,
    "assigned_batch_size": 1,
    "gradient_accumulation_multiplier": 32
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
```

### PPI Evaluation
```python
from ares import ARES

ppi_config = {
    "evaluation_datasets": ["data/unlabeled_eval.tsv"],
    "few_shot_examples_filepath": "data/few_shot_examples.tsv",
    "checkpoints": ["checkpoints/context_relevance.pt"],
    "labels": ["Context_Relevance_Label"],
    "gold_label_path": "data/gold_labels.tsv"
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
# Returns: predicted accuracy with confidence intervals
```

### Full Pipeline
```python
from ares import ARES

# Combine all configs
ares = ARES(
    synthetic_query_generator=synth_config,
    classifier_model=classifier_config,
    ppi=ppi_config
)

# Run complete pipeline
results = ares.run()
```

### Local Model Execution
```python
from ares import ARES

local_config = {
    "model_choice": "meta-llama/Llama-2-13b-hf",
    "vllm": True,
    "host_url": "http://0.0.0.0:8000/v1"
}

ares = ARES(ues_idp=local_config)
results = ares.ues_idp()
```

---

## Configuration Parameters Reference

### Synthetic Generation Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `document_filepaths` | Input document TSV files | Required |
| `few_shot_prompt_filename` | Few-shot examples file | Required |
| `synthetic_queries_filenames` | Output file paths | Required |
| `model_choice` | HuggingFace model | `google/flan-t5-xxl` |
| `documents_sampled` | Number of docs to sample | Required |
| `clean_documents` | Clean docs before gen | `False` |
| `question_temperatures` | Sampling temperatures | `[0.7]` |
| `number_of_negatives_added_ratio` | Negative samples ratio | `0.0` |

### Training Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `classification_dataset` | Training data paths | Required |
| `label_column` | Column(s) to predict | Required |
| `model_choice` | Fine-tune model | `microsoft/deberta-v3-large` |
| `num_epochs` | Training epochs | `10` |
| `patience_value` | Early stopping patience | `3` |
| `learning_rate` | Learning rate | `5e-6` |
| `assigned_batch_size` | Batch size | `1` |
| `gradient_accumulation_multiplier` | Gradient steps | `32` |

### PPI Evaluation Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `evaluation_datasets` | Eval data paths | Required |
| `checkpoints` | Trained model paths | Required |
| `labels` | Label columns | Required |
| `alpha` | Significance level | `0.05` |
| `num_trials` | CI estimation trials | `100` |

---

## Dependencies

```toml
[project]
dependencies = [
    "ares-ai",
    "torch",
    "transformers",
    "openai",
]

[project.optional-dependencies]
local = [
    "vllm",
]
```

---

## System Requirements

- GPU-equipped hardware (for training)
- API keys: OpenAI, TogetherAI, Anthropic (as needed)
- Datasets:
  - Document passages
  - Human validation set (~150 annotated datapoints)
  - Few-shot examples

---

## Resources

- **Documentation**: https://ares-ai.vercel.app/
- **GitHub**: https://github.com/stanford-futuredata/ARES
- **Research Paper**: Stanford Future Data Lab publication
- **Datasets**: KILT, SuperGLUE integrations available
