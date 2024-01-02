import os
import pytest

from coppafish import BuildPDF


@pytest.mark.slow
def test_BuildPDF():
    robominnie_notebook = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../../robominnie/test/integration_dir/output_coppafish/notebook.npz",
    )
    if not os.path.isfile(robominnie_notebook):
        assert False, f"Could not find robominnie notebook at\n\t{robominnie_notebook}.\nRun an integration test first"

    BuildPDF(robominnie_notebook, auto_open=False)
