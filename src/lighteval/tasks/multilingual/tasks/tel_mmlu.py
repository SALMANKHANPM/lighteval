"""
name:
tel_mmlu

dataset:
sarvamai/mmlu-indic

subset:
te

abstract:
mmlu multilingual benchmark.

languages:
telugu

tags:
knowledge, multilingual, multiple-choice

paper:
"""
from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.TELUGU.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.TELUGU,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": line["answer"],
            },
            formulation=formulation,
        ),
        hf_repo="sarvamai/mmlu-indic",
        hf_subset="te",
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
