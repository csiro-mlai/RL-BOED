"""Logger module.

This module instantiates a global logger singleton.
"""
from dowel.histogram import Histogram
from dowel.logger import Logger, LoggerWarning, LogOutput
from dowel.csv_output import CsvOutput  # noqa: I100
from dowel.tensor_board_output import TensorBoardOutput
from pyro.dowel.simple_outputs import StdOutput, TextOutput
from pyro.dowel.tabular_input import TabularInput

logger = Logger()
tabular = TabularInput()

__all__ = [
    'Histogram',
    'Logger',
    'CsvOutput',
    'StdOutput',
    'TextOutput',
    'LogOutput',
    'LoggerWarning',
    'TabularInput',
    'TensorBoardOutput',
    'logger',
    'tabular',
]
