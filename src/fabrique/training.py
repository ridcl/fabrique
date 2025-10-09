import itertools
import sys
from typing import Sized
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm

#############################################################################
#                                   Logger                                  #
#############################################################################


class TrainLogger:
    def log(self, name: str, value: Any): ...
    def update(self): ...


class TqdmLogger(TrainLogger):
    def __init__(self, total=None):
        self._tqdm = tqdm(total=total)
        self.step = 0
        self.prev_step = -1
        self.latest_metrics = {}

    def log(self, name: str, value: Any, step=None):
        # if explicitly provided, override current step
        self.step = step or self.step
        self.latest_metrics[name] = value

    def update(self):
        if self.latest_metrics:
            display_metrics = {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in self.latest_metrics.items()
            }
            self._tqdm.set_postfix(display_metrics)
        # In normal flow, step is incremented by one at every update()
        # so the difference between step and prev_step is 1.
        # However, if step was set manually in log(), we need to adjust
        # tqdm's state accordingly. Note that sometimes it means
        # negative update
        # assert self.step > self.prev_step
        self._tqdm.update(self.step - self.prev_step)
        self.prev_step = self.step
        self.step += 1


#############################################################################
#                               Train Iterator                              #
#############################################################################


def batched(iterable, n):
    """
    Divide an iterable into chunks of n elements.

    Args:
        iterable: Any iterable (list, tuple, string, generator, etc.)
        n: Size of each batch (must be positive integer)

    Yields:
        Tuples containing n elements each (last batch may be shorter)

    Examples:
        >>> list(batched([1, 2, 3, 4, 5, 6, 7], 3))
        [(1, 2, 3), (4, 5, 6), (7,)]

        >>> list(batched("ABCDEFG", 3))
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

        >>> list(batched(range(10), 4))
        [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    """
    if n <= 0:
        raise ValueError("Batch size must be positive")

    iterator = iter(iterable)

    while True:
        batch = tuple(itertools.islice(iterator, n))
        if not batch:
            break
        yield batch


class RestartableIterator:

    def __init__(
        self, base: Callable[[], Any] | Collection, restart_callback: Callable[[], None]
    ):
        self.base = base
        self.restart_callback = restart_callback
        self.state = self._init_state()

    def _init_state(self):
        if isinstance(self.base, Sized):
            it = iter(self.base)
        elif isinstance(self.base, Callable):
            it = self.base()
        else:
            raise ValueError(
                "Base must be either a Collection, or a Callable that returns "
                + f"a fresh iterator, but it is an instance of {type(self.base)} instead"
            )
        return it

    def __iter__(self):
        return self

    def __next__(self):
        try:
            value = next(self.state)
        except StopIteration:
            # run callback; callback may raise StopIteration
            self.restart_callback()
            # restart iterator
            self.state = self._init_state()
            value = next(self.state)
        return value


@dataclass
class TrainIterator:
    base: Callable[[], Any] | Sized
    max_epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: Optional[int] = None
    logger: str | TrainLogger = "tqdm"

    def __post_init__(self):
        # init internal iterator
        def stop_on_max_epochs():
            self.epoch += 1
            if self.epoch >= self.max_epochs:
                raise StopIteration()

        self.ri = RestartableIterator(self.base, restart_callback=stop_on_max_epochs)
        # apply batching if needed
        if self.batch_size:
            self.ri = batched(self.ri, self.batch_size)
        self.epoch: int = 0
        self.step: int = 0
        # init logger
        self.logger = self._init_logger(self.logger)

    def _init_logger(self, logger):
        if isinstance(logger, str):
            if logger == "tqdm":
                if hasattr(self.base, "__len__"):
                    length = len(self.base)
                    bsz = self.batch_size or 1
                    max_steps = self.max_steps or sys.maxsize
                    total = min(length * self.max_epochs // bsz, max_steps)
                else:
                    total = None
                logger = TqdmLogger(total=total)
            else:
                raise ValueError(f"Logger {logger} is not supported")
        return logger

    @property
    def finished_steps(self):
        return self.step

    @property
    def finished_epochs(self):
        return self.epoch

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_steps and self.finished_steps >= self.max_steps:
            raise StopIteration()
        value = next(self.ri)
        self.step += 1
        self.logger.update()
        return value

    def log(self, name: str, value: Any):
        self.logger.log(name, value, step=self.step)
