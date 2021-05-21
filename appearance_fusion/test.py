import os

from config import get_config
from trainer import Trainer


def main():
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join(
            config.logging_root_dir,
            config.experiment_name,
            'checkpoints')

    tester = Trainer(config, test_mode=True)
    tester.test()


if __name__ == '__main__':
    config = get_config()
    main()
