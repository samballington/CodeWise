import textwrap
import pytest
import complexity

@pytest.fixture(autouse=True)
def stub_semantic(monkeypatch):
    # Ensure tests run offline; replace heavy MiniLM logic
    monkeypatch.setattr(complexity, "semantic_complexity_score", lambda txt: 0.1, raising=False)


def assert_valid(text: str):
    size = complexity.choose_chunk_size(text)
    scs = complexity.semantic_complexity_score(text)
    assert isinstance(size, int) and 0 < size <= 1200
    assert isinstance(scs, float)


def test_css():
    css = ".a{color:red;}\n" * 40
    assert_valid(css)


def test_html_js():
    html_js = textwrap.dedent("""
    <html><script>function x() {}</script></html>
    """) * 20
    assert_valid(html_js)


def test_java():
    java_methods = "\n".join([f"void m{i}() {{}}" for i in range(60)])
    java = f"public class D {{\n{java_methods}\n}}"
    assert_valid(java) 