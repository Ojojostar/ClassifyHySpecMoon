from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
import keras
#    stole from this guy https://gist.github.com/Raverss/6b16121be8cf04cd5f26b2806b81890a


def visualize_lr_schedule(opt: keras.optimizers.schedules.LearningRateSchedule, n_steps: int, step_s: int) -> Tuple[List[int], List[float]]:
    """Visualize tensorflow keras scheduler learning rate over the course of n_steps training steps.

    Args:
        opt (LearningRateSchedule): learning rate scheduler to be visualized.
        n_steps (int): number of training steps to visualize over.
        step_s (int): sampling step of the training steps, i.e. takes each step_s in the interval [0, n_steps].

    Returns:
        Tuple[List[int], List[float]]: returns x (steps), y (lr values) values of the plot.
    """
    lr = []
    steps = list(range(0, n_steps, step_s))

    for step in steps:
        lr_at_s = opt(step).numpy()
        lr.append(lr_at_s)

    plt.suptitle(f'Learning rate schedule of {type(opt).__name__}')
    plt.plot(steps, lr)

    return steps, lr