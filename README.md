# ncm-lab

How much faster is Next-Concept Modeling (NCM) than a same-size autoregressive model on an
RTX 3050 (6 GB), at what quality, and does that speed matter inside a real RAG or agent pipeline?

## Phases

| Phase | Topic | Status |
|---|---|---|
| 1 | Python and PyTorch foundations | in progress |
| 2 | Toy NCM from scratch (`src/ncm_lab/toy_ncm`) | |
| 3 | Official NCM checkpoints (`src/ncm_lab/models`) | |
| 4 | Fair benchmark of the speed claim (`src/ncm_lab/bench`) | |
| 5 | NCM inside RAG and agents (`src/ncm_lab/rag`, `agents`) | |
| 6 | Production service (`src/ncm_lab/api`) | |

Each finished phase is tagged `phase-N`. Notes per phase live in `docs/`.

## Setup (WSL2 Ubuntu)

```bash
git clone https://github.com/Aymenrahmanii/NCM.git ncm-lab && cd ncm-lab
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
cp .env.example .env
```

Model weights (`artifacts/`), datasets (`data/`) and the official code (`third_party/ncm`)
are not in Git; download scripts in `scripts/` recreate them.

## Results

Benchmark CSVs and charts go in `results/`.

## Limitations

To be written honestly in Phase 6.
