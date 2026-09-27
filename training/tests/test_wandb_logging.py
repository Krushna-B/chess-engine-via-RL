import unittest

from chess_training.wandb_logging import WandbLogger


class WandbLoggingTest(unittest.TestCase):
    def test_disabled_logger_is_a_noop(self):
        logger = WandbLogger(False, {"test": True})
        logger.log({"metric": 1.0}, step=1)
        logger.finish()


if __name__ == "__main__":
    unittest.main()
