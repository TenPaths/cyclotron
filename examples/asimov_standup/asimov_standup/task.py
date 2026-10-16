"""Task registration helpers."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from .config import ppo_runner_cfg, standup_env_cfg
from .runner import StandupRunner

TASK_ID = "Mjlab-Standup-Asimov-1"


def register_task() -> None:
    """Register once in mjlab's process-local task registry."""
    if TASK_ID in list_tasks():
        return
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=standup_env_cfg(),
        play_env_cfg=standup_env_cfg(play=True),
        rl_cfg=ppo_runner_cfg(),
        runner_cls=StandupRunner,
    )


register_task()
