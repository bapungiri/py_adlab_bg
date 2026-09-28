"""Dataframe builders for group analyses.

Each builder is a per-subject function decorated with 'group_builder', which
handles looping over exps, attaching 'exp.common_kwargs', concatenating and
saving to GroupData (with metadata). Function names match GroupData basenames.
"""

from ._core import group_builder
