import torch
import hydra
import logging
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.utils import instantiate

load_dotenv('.env')
log = logging.getLogger(__name__)


def validate_data_formats(cfg):
      if not set(cfg.algorithm.data_formats).intersection(set(cfg.model.data_formats), set(cfg.datamodule.data_formats)):
            raise NotImplementedError("Incompatible data formats in algorithm, model, and datamodule configurations.")
      
 
@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main(cfg):

      log.info(OmegaConf.to_yaml(cfg))
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "flexible" else cfg.device

      validate_data_formats(cfg)

      model = instantiate(cfg.model.network_config, 
                          loss = cfg.loss, 
                          optimizer_config = cfg.optimizer,
                          scheduler_config = cfg.scheduler
                          ).to(device)
      
      datamodule = instantiate(cfg.datamodule.data_config)
      
      trainer = instantiate(cfg.algorithm.trainer, 
                            model, 
                            datamodule, 
                            device, 
                            cfg)
      trainer.fit()

if __name__ == "__main__":
      main()