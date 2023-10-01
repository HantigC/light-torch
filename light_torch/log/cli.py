from .base import EpochLogBase


class CliLog(EpochLogBase):
    def log_at_epoch(self, what, epoch_num, stage):
        print(f"STAGE {stage}: Epoch: {epoch_num: 04d}")
        print(what)
