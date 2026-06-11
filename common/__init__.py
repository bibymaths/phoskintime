"""Shared PhosKinTime utilities."""


def format_duration(seconds):
    """Format a duration in seconds without importing heavy common utilities at package import time."""
    from common.utils.display import format_duration as _format_duration

    return _format_duration(seconds)
