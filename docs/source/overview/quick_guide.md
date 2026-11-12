# Quick guide

Parsers turn an annotated corpus into an `EvaluationDataset`. Metrics are computed by comparing a gold dataset with a predicted one.

Currently implemented metrics: top-N accuracy (micro and macro).

Each corpus has its own parser under `usas_evaluation_framework.parsers`, all subclassing `BaseParser`. See the API reference for details.
