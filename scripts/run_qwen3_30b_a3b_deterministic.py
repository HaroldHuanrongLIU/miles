"""True-on-policy deterministic variant of run_qwen3_30b_a3b.

Extends the base script with true-on-policy contract, EP parity defaults,
and deterministic launch plan injection. Use this script when you need
exact-zero logprob alignment between SGLang rollout and Megatron training.
"""

import os
from dataclasses import dataclass

import typer
from scripts.run_qwen3_30b_a3b import ScriptArgs as BaseScriptArgs
from scripts.run_qwen3_30b_a3b import prepare

import miles.utils.external_utils.command_utils as U
from miles.true_on_policy import apply_true_on_policy_script_defaults, build_true_on_policy_launch_plan


@dataclass
class ScriptArgs(BaseScriptArgs):
    true_on_policy: bool = True
    true_on_policy_contract: str | None = None
    true_on_policy_fast_decode: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.sglang_expert_parallel_size == 1 and self.true_on_policy:
            self.sglang_expert_parallel_size = self.expert_model_parallel_size
        if (
            self.true_on_policy
            and self.sglang_expert_parallel_size > 1
            and self.rollout_num_gpus_per_engine < self.sglang_expert_parallel_size
        ):
            self.rollout_num_gpus_per_engine = self.sglang_expert_parallel_size
        apply_true_on_policy_script_defaults(self)


def execute(args: ScriptArgs):
    from scripts.run_qwen3_30b_a3b import execute as base_execute

    plan = build_true_on_policy_launch_plan(args)
    os.environ.update(plan.env_vars)
    args.extra_args = f"{plan.train_args} {args.extra_args}"
    base_execute(args)


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
