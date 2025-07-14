import importlib, sys
from langchain.globals import get_llm_cache
from langchain.cache import SQLiteCache, InMemoryCache


def _reload_agent():
    if 'agent' in sys.modules:
        importlib.reload(sys.modules['agent'])
    else:
        importlib.import_module('agent')


def test_sqlite_cache_selected(monkeypatch):
    monkeypatch.setenv('LC_CACHE', 'sqlite')
    _reload_agent()
    assert isinstance(get_llm_cache(), SQLiteCache)


def test_inmemory_cache_fallback(monkeypatch):
    monkeypatch.setenv('LC_CACHE', 'invalid_scheme')
    _reload_agent()
    assert isinstance(get_llm_cache(), InMemoryCache) 