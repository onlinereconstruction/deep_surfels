import os

from config import get_config
from trainer import Trainer


def main():
    trainer = Trainer(config)
    trainer.train()

    config.checkpoint_path = os.path.join(
        config.logging_root_dir,
        config.experiment_name,
        'checkpoints')

    # evaluation
    tester = Trainer(config, test_mode=True)
    tester.test()


if __name__ == '__main__':
    config = get_config()
    main()
