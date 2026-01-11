# HRM
git clone https://github.com/sapientinc/HRM.git
cd HRM

# uv + env
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.10 .venv
source .venv/bin/activate    # or skip and always use `uv run`

# tools
sudo apt update && sudo apt install -y build-essential git ninja-build python3-dev

# CUDA 12.6 side-by-side
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
wget -q --show-progress -O cuda_12_6.run $CUDA_URL
sudo sh cuda_12_6.run --silent --toolkit --override
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# PyTorch cu126 + helpers
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
uv pip install --index-url $PYTORCH_INDEX_URL torch torchvision torchaudio
uv pip install packaging ninja wheel setuptools setuptools-scm

# FlashAttention (pick ONE)
# Hopper → FA3 from source
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/hopper && uv run python setup.py install && cd ../../
# Ampere/Ada → FA2 wheel
uv pip install setuptools
uv pip install psutil
uv pip install flash-attn


git submodule update --init --recursive
uv pip install -r requirements.txt

# demo
 uv run python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
# OMP_NUM_THREADS=8 uv run python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
nohup bash -c 'OMP_NUM_THREADS=8 uv run python evaluate.py \
checkpoint=Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt \
save_outputs="[\"inputs\",\"labels\",\"puzzle_identifiers\",\"logits\",\"q_halt_logits\",\"q_continue_logits\",\"intermediate_preds\"]"'   > eval.log 2>&1 &

tail -f eval.log

