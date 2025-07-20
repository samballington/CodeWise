import textwrap
from complexity import choose_chunk_size, semantic_complexity_score

def test_homogeneous_text():
    css = ".a{color:red;}\n"*40
    assert semantic_complexity_score(css) < 0.05
    assert choose_chunk_size(css) >= 1000

def test_mixed():
    html_js = textwrap.dedent("""
    <html><body><script>function x(){console.log(1)}</script></body></html>
    """)*20
    scs = semantic_complexity_score(html_js)
    assert 0.05 <= scs < 0.15
    size = choose_chunk_size(html_js)
    assert 400 < size <= 600

def test_complex():
    java = "public class D{\n" + "\n".join([f"void m{i}(){{}}" for i in range(60)]) + "\n}"
    assert semantic_complexity_score(java) >= 0.15
    assert choose_chunk_size(java) == 300 