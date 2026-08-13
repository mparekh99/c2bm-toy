import random
import numpy as np
import torch
import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
import pickle
# import jpype
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

# pytorch lightning
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# data loading
from src.data.dataset_block import get_dataset

# causal discovery
from src.causal_discovery.causal_discovery_block import causal_discovery

# graph completion block
from src.completion.completion_block import complete_graph_with_llm

# training and utils
from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams
from src.data.utils import static_graph_collate
from src.metrics import hamming_distance
from src.plots import maybe_plot_graph
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph
from src.utils import finetune_model


torch.set_printoptions(
    threshold=float("inf"),
    linewidth=1000,
    sci_mode=False,
)


# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
    
def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.mkdir('results')
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # checkpoint_path = os.path.join(
    #     get_original_cwd(),
    #     "outputs",
    #     "multirun",
    #     "2026-08-11",
    #     "rerun_toy",
    #     "0",
    #     "checkpoints",
    #     "epoch=42-step=602.ckpt",
    # )

    checkpoint_path = os.path.join(
        get_original_cwd(),
        "outputs",
        "multirun",
        "2026-08-11",
        "rerun_toy",
        "1",
        "checkpoints",
        "last.ckpt",
    )

# outputs/multirun/2026-08-05/asia_true_graph/1/checkpoints/last.ckpt


# outputs/multirun/2026-08-06/alarm_true_graph

    # print("Loading checkpoint:", checkpoint_path)
    # print("Exists:", os.path.exists(checkpoint_path))

    # aprint(dataset)djust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg)

    # print("=== after get_dataset ===", flush=True)
    # print("load_true_graph:", cfg.dataset.load_true_graph, flush=True)
    # print("load_graph:", cfg.dataset.load_graph, flush=True)
    # print("causal_discovery:", cfg.causal_discovery, flush=True)
    # print("dataset_directory:", dataset_directory, flush=True)

    # print("loading true graph", flush=True)
    graph = true_graph

    graph, dataset = remove_problematic_edges(graph, dataset)

    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1

    # (part 2): remove cycles
    graph = remove_cycles(graph, y_index)

    # use the graph to define an intervention policy at test time
    interv_policy, ip_names = get_intervention_policy(cfg.policy, graph, true_graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    cfg = update_config_from_data(cfg, dataset)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)

    [dataset.data[split].register_graph(graph) for split in dataset.data]

    test_dataloader = DataLoader(dataset.data['test'], 
                                batch_size=cfg.dataset.batch_size, 
                                collate_fn=static_graph_collate,
                                num_workers=cfg.dataset.num_workers)

    engine = instantiate(cfg.engine)

    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))

        trainer.test(engine, test_dataloader, ckpt_path=checkpoint_path)
        trainer.logger.finalize("success")

    finally:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

if __name__ == "__main__":
    main()
    print('done')