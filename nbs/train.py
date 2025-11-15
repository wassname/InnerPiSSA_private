from ipissa.train.train_adapter import main
from ipissa.config import proj_root, TrainingConfig, default_configs
import tyro 

if __name__ == "__main__":
    # config = tyro.cli(TrainingConfig, use_underscores=True)
    # if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_configs, use_underscores=True)
    main(config)
