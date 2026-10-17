"""The Streamlit app must actually render (a syntax/import error would otherwise show up as an empty page)."""
import os

import pytest
from streamlit.testing.v1 import AppTest


def run_app():
    at = AppTest.from_file("streamlit_app.py", default_timeout=180).run()
    assert not at.exception, [e.message for e in at.exception]
    assert [t.value for t in at.title], "app rendered no title (script failed before rendering)"
    return at


def test_nowcast_view_renders_all_models():
    at = run_app()
    assert len(at.dataframe[0].value) == 9          # 3 models x 3 pollutants


@pytest.mark.parametrize("task,h", [("Day 1 ahead (+24 h)", 24), ("Day 2 ahead (+48 h)", 48), ("Day 3 ahead (+72 h)", 72)])
def test_forecast_views_render(task, h):
    if not os.path.exists(f"weights/trained_neural_ode_model_h{h}.pth"):
        pytest.skip("forecast models not trained")
    at = run_app()
    at.sidebar.selectbox[0].select(task).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.dataframe[0].value) == 9
