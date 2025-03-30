def location(path: str, label: str = None) -> str:
    """
    Returns a clickable hyperlink string for supported terminals using ANSI escape sequences.

    Args:
        path (str): The file path or URL.
        label (str, optional): The display text for the link. Defaults to the path if not provided.

    Returns:
        str: A string that, when printed, shows a clickable link in terminals that support ANSI hyperlinks.
    """
    if label is None:
        label = path
    # Ensure the path is a URL (for file paths, prepend file://)
    if not (path.startswith("http://") or path.startswith("https://") or path.startswith("file://")):
        path = f"file://{path}"
    # ANSI escape sequence format: ESC ] 8 ; ; <URL> ESC \ <label> ESC ] 8 ; ; ESC \
    return f"\033]8;;{path}\033\\{label}\033]8;;\033\\"