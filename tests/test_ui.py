import numpy as np

import ui


def test_card_escapes_user_text():
    h = ui.card('<script>x</script>', "happy", 0.5, None, truth="<b>sad</b>")
    assert "<script>" not in h and "&lt;script&gt;" in h
    assert "<b>sad" not in h


def test_low_confidence_badge_and_alt():
    assert "low" in ui.card("a", "sad", 0.1, None, None, ("happy", 0.2))
    assert "next likely" in ui.card("a", "sad", 0.9, None, None, ("happy", 0.2))
    assert 'class="low"' not in ui.card("a", "sad", 0.9)


def test_heatmap_renders_every_cell():
    s = np.random.RandomState(0).randn(10, 10)
    h = ui.heatmap_html(s, [f"r{i}" for i in range(10)], ["l%d" % i for i in range(10)])
    assert h.count("<td") == 100


def test_footer_has_credit_link():
    assert "linkedin.com/in/gaurav-tomar-630b2a316" in ui.footer_html()
