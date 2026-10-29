import pytest

from tests.conftest import ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PAGES = ["app/app.py", "app/pages/1_Price_Predictor.py", "app/pages/2_Analytics_Dashboard.py",
         "app/pages/3_Recommend_Properties.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_loads(page):
    app = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not app.exception


def test_predict_button_shows_estimate_and_range():
    app = AppTest.from_file(str(ROOT / "app/pages/1_Price_Predictor.py"), default_timeout=120).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Estimated price" in s.value for s in app.success)
    assert any("Likely range" in m.value for m in app.markdown)
