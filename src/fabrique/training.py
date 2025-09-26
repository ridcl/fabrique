from typing import Optional, Callable, Any
from dataclasses import dataclass

@dataclass
class EpochLoop:  # with_epochs?
    iter_fn: Callable[[], Any]
    max_epochs: int
    max_steps: Optional[int] = None

    def __post_init__(self):
        self.epoch: int = 0
        self.step: int = 0
        self.state = self.iter_fn()

    def __iter__(self):
        return self

    def __next__(self):
        # if max steps or max_epochs and iter end -> StopIteration()
        try:
            value = next(self.state)
        except StopIteration:
            if self.epoch == self.max_epochs:
                raise
            else:
                # restart iterator
                self.state = self.iter_fn()
                value = next(self.state)
        # increment counters
        # advance pbar
        return value