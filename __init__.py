from __future__ import annotations

__version__ = "0.4.0"
__author__ = "Abhinav Mishra"
__email__ = "mishraabhinav36@gmail.com"
__license__ = "BSD-3-Clause"
__description__ = "A Python package for modeling and optimizing kinase-phosphorylation dynamics in triple negative breast cancer."
__url__ = "https://github.com/bibymaths/phoskintime"
__project_name__ = "phoskintime"


def main(*args, **kwargs):
    from protwise.runner.main import main as _main
    return _main(*args, **kwargs)


__all__ = ["main", "__version__", "__author__", "__email__", "__license__", "__description__", "__url__", "__project_name__"]
