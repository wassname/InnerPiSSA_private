from repeng.train.train_adapter import main, TrainingConfig
import tyro 

if __name__ == "__main__":
    config = tyro.cli(TrainingConfig, use_underscores=True)
    main(config)
