"""
Exception classes for formula search module.

Based on TangentCFT (Tangent Combined FastText)
Original repository: https://github.com/BehroozMansouri/TangentCFT
Original authors: Nidhin Pattaniyil, Richard Zanibbi
Adapted for multirag formula_search module
"""

__author__ = 'Nidhin'


class UnknownTagException(Exception):
    """
    An exception to indicate unknown Mathml tag
    """

    def __init__(self, tag):
        self.tag = tag
