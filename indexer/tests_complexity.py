import textwrap
from complexity import choose_chunk_size, semantic_complexity_score

def test_homogeneous_text():
    css = ".container { display: flex; }\n" * 40
    scs = semantic_complexity_score(css)
    assert scs < 0.05
    assert choose_chunk_size(css) >= 1000

def test_mixed_html_js():
    mixed = textwrap.dedent(
        """
        <html><head><title>Test</title></head>
        <body>
        <script>
        function hello() { console.log('hi'); }
        </script>
        </body>
        </html>
        """
    ) * 20
    scs = semantic_complexity_score(mixed)
    assert 0.05 <= scs < 0.15
    size = choose_chunk_size(mixed)
    assert 400 < size <= 600

def test_complex_controller():
    java_methods = "\n".join([
        f"    public void m{i}() {{ System.out.println({i}); }}" for i in range(60)
    ])
    java = f"public class Demo {{\n{java_methods}\n}}"
    scs = semantic_complexity_score(java)
    assert scs >= 0.15
    assert choose_chunk_size(java) == 300 