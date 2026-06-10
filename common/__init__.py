def format_duration(seconds):
    """Format a duration in seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} sec"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} hr"
