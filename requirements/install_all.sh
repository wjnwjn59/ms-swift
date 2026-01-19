# please use python=3.10/3.11, cuda12.*
# sh requirements/install_all.sh
pip install uv
uv pip install "sglang<0.5.6" -U
uv pip install "vllm>=0.5.1,<0.11.1" -U
uv pip install "lmdeploy>=0.5,<0.10.2" -U
uv pip install "transformers<4.58" "trl<0.25" peft -U
uv pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
uv pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
uv pip install timm "deepspeed<0.18" -U
uv pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
uv pip install decord librosa icecream soundfile -U
uv pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
