"""Route a study run to experiments with one arm and evaluation configuration.

The run group is checkpoint metadata, not a Braintrust experiment. Repeats within
one stage/configuration remain together. Creating/logging here never runs a model.
"""
from dataclasses import dataclass
import re
import threading
from types import SimpleNamespace
from urllib.parse import quote

from run.model_metadata import experiment_model_metadata, model_metadata


@dataclass(frozen=True)
class Configuration:
    arm: str
    target: str
    scheme: str
    rubric: str

    def metadata(self):
        return {**experiment_model_metadata([self.arm]), 'arm': self.arm,
                'target': self.target, 'scheme': self.scheme, 'rubric': self.rubric,
                'experiment_layout': 'single_arm_configuration_v1'}

    def name(self, run_group):
        values = (self.arm, self.target, self.scheme, self.rubric)
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in values):
            raise ValueError('Configuration names must be explicit stable identifiers')
        return run_group + '__' + '__'.join(values)


def configuration(metadata):
    arm = metadata['arm']
    model_metadata(arm)  # Fail before logging/inference for an unknown arm.
    rubric = metadata.get('rubric', 'native')
    return Configuration(arm, metadata.get('target', 'interaction_harm'),
                         metadata.get('scheme', 's3' if rubric == 'native_s3' else 's2'), rubric)


class ExperimentGroup:
    def __init__(self, *, project=None, project_id=None, experiment, metadata=None,
                 update=False, set_current=False, **kwargs):
        if set_current:
            raise ValueError('Experiment groups cannot be the current span parent')
        self.project = project
        self.project_id = project_id
        self.name = experiment
        self.metadata = dict(metadata or {})
        self.update = update
        self.kwargs = kwargs
        self.children = {}
        self.experiment_urls = {}
        self.lock = threading.Lock()

    def for_configuration(self, config):
        import braintrust as bt
        with self.lock:
            if config not in self.children:
                metadata = {**self.metadata, **config.metadata(), 'run_group': self.name}
                child = bt.init(project=self.project, project_id=self.project_id,
                                experiment=config.name(self.name), metadata=metadata,
                                update=self.update, set_current=False, **self.kwargs)
                self.children[config] = child
                # Deterministic URL avoids a summarize request on every first span.
                self.experiment_urls[config.name(self.name)] = (
                    'https://www.braintrust.dev/app/eval_engineering/p/'
                    + quote(self.project or 'decision-interface-safety', safe='')
                    + '/experiments/' + quote(config.name(self.name), safe=''))
            return self.children[config]

    def _route(self, event):
        metadata = {**{k: self.metadata[k] for k in ('arm', 'target', 'scheme', 'rubric') if k in self.metadata},
                    **(event.get('metadata') or {})}
        config = configuration(metadata)
        event['metadata'] = {**metadata, **model_metadata(config.arm),
                             'target': config.target, 'scheme': config.scheme,
                             'rubric': config.rubric, 'run_group': self.name}
        return self.for_configuration(config)

    def start_span(self, **event):
        return self._route(event).start_span(**event)

    def log(self, **event):
        return self._route(event).log(**event)

    def flush(self):
        for child in list(self.children.values()):
            child.flush()

    def summarize(self):
        # Kept for runner reports; checkpoints also contain exact child URLs.
        return SimpleNamespace(experiment_url=(
            'https://www.braintrust.dev/app/eval_engineering/p/'
            + quote(self.project or 'decision-interface-safety', safe='')
            + '/experiments'))


def init_experiments(**kwargs):
    return ExperimentGroup(**kwargs)
