"""N6 (option C): posterior sampling is disabled; requesting it is a clear error pointing to N6."""
from types import SimpleNamespace

import pytest


def test_default_is_off_and_passes():
    from networkmodel.runner import check_posterior_disabled

    check_posterior_disabled(SimpleNamespace(posterior_sampling=False))


def test_request_is_refused_with_n6_message():
    from networkmodel.runner import check_posterior_disabled

    with pytest.raises(ValueError, match="N6"):
        check_posterior_disabled(SimpleNamespace(posterior_sampling=True))
