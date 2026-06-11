from abc import ABC, abstractmethod
from pathlib import Path

from usas_evaluation_framework.dataset import EvaluationDataset


class BaseParser(ABC):
    @staticmethod
    @abstractmethod
    def parse(dataset_path: Path,
              label_validation: set[str] | None = None,
              label_filter: set[str] | None = None,
              dataset_name: str | None = None,
              language: str | None = None,
              ) -> EvaluationDataset:
        """
        Parse the given dataset path and return an EvaluationDataset.

        Args:
            dataset_path: The path to the dataset to parse.
            label_validation: A set of labels that the semantic/dataset labels should
                be validated against. Defaults to `None` in which case no validation
                is performed.
            label_filter: A set of labels from the dataset that should be filtered out.
                Defaults to `None` in which case no filtering is performed.
            dataset_name: Name for the returned dataset. Defaults to `None` in which
                case each concrete parser can supply its own default.
            language: Language of the corpus (e.g. ``'Spanish'``). Defaults to `None`.

        Returns:
            EvaluationDataset: The parsed dataset.
        """