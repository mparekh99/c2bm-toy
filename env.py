from os import environ as env
from pathlib import Path
import os

SLATE = "/N/slate/mihparek"

os.environ["HF_HOME"] = f"{SLATE}/hf_cache"
os.environ["TORCH_HOME"] = f"{SLATE}/torch_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{SLATE}/hf_cache"
os.environ["HF_DATASETS_CACHE"] = f"{SLATE}/hf_cache/datasets"
env["TOKENIZERS_PARALLELISM"] = "false"
env['HYDRA_FULL_ERROR'] = '1'

PROJECT_NAME = "c2bm"
WANDB_ENTITY = "mihparek-university-of-indiana-bloomington"

CACHE = Path("/N/slate/mihparek/c2bm_cache")
CACHE.mkdir(exist_ok=True)

ROBOT_MUG_PATH = Path("/N/slate/mihparek/Fail2/evals_1.5/target/CoffeeSetupMug/Masked")

HUGGINGFACEHUB_TOKEN='REMOVED_HF_TOKEN'
OPENAI_API_KEY=''