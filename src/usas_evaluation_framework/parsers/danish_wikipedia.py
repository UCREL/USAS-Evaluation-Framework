from pathlib import Path

from usas_evaluation_framework.dataset import EvaluationDataset
from usas_evaluation_framework.parsers.wikipedia import WikipediaUSAS


class DanishWikipediaUSAS(WikipediaUSAS):
    """
    Convenience subclass of :class:`WikipediaUSAS` pre-configured for the
    Danish medical Wikipedia USAS corpus.

    Calling :meth:`parse` without ``dataset_name`` or ``language`` arguments
    uses ``dataset_name='Medical Wikipedia'`` and ``language='Danish'``.
    Both defaults can be overridden by the caller.
    """

    @staticmethod
    def parse(
        dataset_path: Path,
        label_validation: set[str] | None = None,
        label_filter: set[str] | None = None,
        dataset_name: str | None = "Medical Wikipedia",
        language: str | None = "Danish",
    ) -> EvaluationDataset:
        return WikipediaUSAS.parse(
            dataset_path,
            label_validation=label_validation,
            label_filter=label_filter,
            dataset_name=dataset_name,
            language=language,
        )
