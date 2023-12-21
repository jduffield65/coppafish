import os
import pytest

from coppafish import Notebook, BuildPDF


@pytest.mark.slow
def test_BuildPDF():
    robominnie_notebook = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../../robominnie/test/integration_dir/output_coppafish/notebook.npz",
    )
    if not os.path.isfile(robominnie_notebook):
        assert False, f"Could not find robominnie notebook at\n\t{robominnie_notebook}\nRun an integration test first"

    # nb = Notebook(robominnie_notebook)
    # BuildPDF(nb)

    nb = Notebook("/home/paul/Documents/coppafish/dante/output/notebook.npz")
    BuildPDF(
        nb,
        output_path="/home/paul/Documents/coppafish/my_branch/coppafish/coppafish/robominnie/test/integration_dir/output_coppafish/diagnostics.pdf",
    )


test_BuildPDF()
