"""Expose the package version metadata and delegate CLI entry-point calls to networkmodel.runner; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.runner."""
__version__ = "0.1.0"
__author__ = "Abhinav Mishra"
__email__ = "mishraabhinav36@gmail.com"


def main(*args, **kwargs):
    """Run the networkmodel entry point
    
    Args:
        args: Positional arguments forwarded to the runner.
        kwargs: Keyword arguments forwarded to the runner.
    
    Returns:
        Computed result from this routine.
    """
    from networkmodel.runner import main as _main
    return _main(*args, **kwargs)


app = main
