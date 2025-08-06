#!/usr/bin/env python3
"""
Simple Cache Test - Validate path resolution cache works
"""

def test_cache():
    print("Testing path resolution cache...")
    
    try:
        from path_resolution_cache import PathResolutionCache, PathResolutionCacheConfig
        print("Successfully imported cache modules")
        
        # Create cache
        config = PathResolutionCacheConfig(max_size=10, ttl_seconds=60)
        cache = PathResolutionCache(config)
        print("Created cache instance")
        
        # Test put/get
        cache.put("test/file", "/resolved/test/file", True)
        result = cache.get("test/file")
        print(f"Put/Get test: {result}")
        
        if result == ("/resolved/test/file", True):
            print("PASS: Basic cache operations work")
        else:
            print(f"FAIL: Expected ('/resolved/test/file', True), got {result}")
            return False
        
        # Test stats
        stats = cache.get_stats()
        print(f"Cache stats: hits={stats.get('hits', 0)}, total={stats.get('total_requests', 0)}")
        
        print("Testing PathResolver integration...")
        from path_resolver import PathResolver
        
        # Create resolver
        resolver = PathResolver("C:\\Users\\suga\\Desktop\\aistuff\\codewise\\workspace")
        print("Created PathResolver with cache")
        
        # Test path resolution
        result = resolver.resolve_file_path("README.md")
        print(f"Path resolution result: {result}")
        
        print("SUCCESS: Path resolution cache is working!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cache()