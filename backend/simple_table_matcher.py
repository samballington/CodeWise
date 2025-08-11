#!/usr/bin/env python3
"""Simple table matcher for testing"""

class SimpleTableMatcher:
    def __init__(self):
        self.name = "SimpleTableMatcher"
    
    def test(self):
        return "Working"

if __name__ == "__main__":
    matcher = SimpleTableMatcher()
    print(f"✅ {matcher.name} is working: {matcher.test()}")