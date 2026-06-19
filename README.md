# VLM Cytomorphology Eval

Automated evaluation framework for vision-language models (VLMs) on cytology tasks and datasets.

## Repository layout

```
.
├── src/vlm_cytomorphology_eval/
│   ├── config/             # dataset paths, naming conventions
│   ├── prompts/            # task-specific prompts per dataset
│   ├── models/             # API wrappers (GPT-4o, Gemini, Llama, DeepSeek, LLaVA-Med, MedFlamingo)
│   ├── data/               # dataset preparation and fold splitting
│   ├── inference/          # main entry point for VLM inquiries
│   ├── finetuning/         # DINO / Llama fine-tuning
│   ├── evaluation/         # answer review pipelines
│   ├── features/           # feature extraction
│   ├── visualization/      # confusion matrices, explainability plots
│   ├── expert_review/      # selecting images for human review
│   └── utils/
├── notebooks/              # BiomedCLIP, CONCH zero-shot experiments
├── scripts/                # SLURM / shell launchers
├── tests/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── LICENSE
```

## Installation

We recommend **conda** for this project because two of the supported models pin
incompatible CUDA / PyTorch / `transformers` versions and must live in their
own environments. Plain `venv` works only for the commercial-API-only path
(GPT-4o, Gemini); it cannot manage per-env CUDA toolkits.

You will typically end up with three environments:

| Env | Python | Purpose |
|---|---|---|
| `vlm_eval`  | 3.10 | Default — commercial APIs, Llama, fine-tuning, plotting, notebooks |
| `llavamed`  | 3.10 | LLaVA-Med only (pins older `transformers`) |
| `deepseek`  | 3.9  | DeepSeek-VL2 only (requires CUDA 11.8 + PyTorch 2.0.1) |

### Default environment (`vlm_eval`)

```bash
git clone https://github.com/<your-org>/vlm_cytomorphology_eval.git
cd vlm_cytomorphology_eval

conda create -n vlm_eval python=3.10 -y
conda activate vlm_eval

pip install -e .
```

This covers GPT-4o, Gemini, Llama, BiomedCLIP, CONCH, fine-tuning (DINO),
plotting, and all data-preparation scripts.

### Lightweight alternative (API models only, no GPU)

If you only need the commercial APIs and don't want conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### LLaVA-Med and DeepSeek-VL2

Both require dedicated conda environments installed from source — see the
**Setting up LLaVA-Med** and **Setting up DeepSeek VL2** sections below.

### Setting up LLaVA-Med

LLaVA-Med must be installed from the official Microsoft repository. We recommend a separate conda env because LLaVA-Med pins older versions of `transformers` and `torch`.

```bash
conda create -n llavamed python=3.10 -y
conda activate llavamed

git clone https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med
pip install -e .
```

Pre-trained weights (e.g. `llava-med-v1.5-mistral-7b`) are pulled from HuggingFace at runtime — make sure `HF_TOKEN` is set if the checkpoint is gated.

After installing LLaVA-Med, install this repo's requirements into the same env:

```bash
cd /path/to/vlm_cytomorphology_eval
pip install -e .
```

### Setting up DeepSeek VL2

DeepSeek VL2 requires CUDA 11.8 and PyTorch 2.0.1, so use a dedicated conda env.

```bash
# Check available CUDA versions
ls /usr/local/ | grep cuda

# Activate CUDA 11.8 (adjust paths for your system)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
nvcc --version

# Create and activate a Python 3.9 env
conda create -n deepseek python=3.9 -y
conda activate deepseek

# Install DeepSeek-VL2 from source
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
cd DeepSeek-VL2
pip install -e .

# DeepSeek's requirements omit a few packages this repo needs:
pip install numpy==1.26.4 pandas==1.5.3 openpyxl==3.1.2

# Install this repo into the same env
cd /path/to/vlm_cytomorphology_eval
pip install -e .
```

DeepSeek VL2 only runs on an A100 with 80 GB memory.

### Setting up MedFlamingo (optional)

MedFlamingo uses `open_flamingo` and requires the LLaMA-7B base weights. Follow the [med-flamingo](https://github.com/snap-stanford/med-flamingo) repo's setup.

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | GPT-4o inference and fine-tuning |
| `GEMINI_API_KEY` | Gemini inference |
| `HF_TOKEN` | Downloading gated HuggingFace checkpoints |
| `VLM_ROOT_FOLDER_PATH` | Override default data/results root |

> **Important:** The dataset and results paths inside `src/vlm_cytomorphology_eval/config/dataset_info_and_paths.py` are currently hard-coded to internal cluster locations (`/lustre/...`) and contributor home directories. Edit `LOCAL_ROOT_FOLDER_PATH` and the per-dataset paths to match your setup before running.

## Usage

### Running a VLM evaluation

```bash
python -m vlm_cytomorphology_eval.inference.run_api_inquiry \
    --dataset_name <name> --task_type <task> --vlm_name <model>
```

Make sure the relevant API key is exported (or present in your `.env`).

### Generating reports

```bash
python -m vlm_cytomorphology_eval.visualization.plot_conf_mat
```

This computes confusion matrices and aggregate metrics across all configured models and datasets.

### Adding a new dataset

1. Add the dataset entry to `get_dataset_info` in `src/vlm_cytomorphology_eval/config/dataset_info_and_paths.py`.
2. Add the dataset-specific prompts in `src/vlm_cytomorphology_eval/prompts/prompts.py`.
3. Build the evaluation subset:

   ```bash
   python -m vlm_cytomorphology_eval.data.prepare_dataset \
       --dataset_name <name> --n_per_label <n>
   ```

### Adding a new VLM model family

1. Add the family name to `available_model_families` and choose a representative model in `recommended_models` (both in `get_global_info`).
2. Register the review model in `get_review_model`.
3. Add a new `<family>_api.py` under `src/vlm_cytomorphology_eval/models/` exposing `<family>_api_visual_inquiry`, `<family>_api_text_inquiry`, and any task-specific functions (multi-image, etc.). See `gpt_api.py` for the simplest reference.
4. Wire it into `import_model` in `src/vlm_cytomorphology_eval/inference/run_api_inquiry.py`.

### Adding a new task

1. Append the task name to `available_task_types` in `get_global_info`.
2. Add prompts for all datasets and for both reviewed and unreviewed cases in `prompts.py`.
3. Implement `run_api_<task>` inside `run_api_inquiry.py`.
4. Add any per-model task functions in the relevant model file under `models/`.

## Generated folder structure

The pipeline assumes / creates the following layout under the configured root:

```
<root>/
├── Datasets/<dataset_name>/{train,val,test}/...
├── Results/
│   └── <dataset_name>/
│       ├── answers/
│       └── confusion_matrices/
└── Plots/<dataset_name>/
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the associated publication (link will be added on release).
