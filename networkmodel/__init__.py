__version__ = "0.1.0"
__author__ = "Abhinav Mishra"
__email__ = "mishraabhinav36@gmail.com"


def main(*args, **kwargs):
    from networkmodel.runner import main as _main
    return _main(*args, **kwargs)


app = main
