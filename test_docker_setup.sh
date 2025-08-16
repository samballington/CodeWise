#!/bin/bash

# Docker Test Setup Script
# Tests all Phase 2 and Phase 3 implementations in Docker environment

echo "============================================================"
echo " CodeWise Docker Environment Test"
echo "============================================================"

# Set error handling
set -e

# Function to print status
print_status() {
    echo "--- $1 ---"
}

print_status "Starting Docker Test Environment"

# Check if we're in Docker or need to start Docker
if [ -f /.dockerenv ]; then
    echo "[PASS] Running inside Docker container"
else
    echo "[WARNING] Not in Docker container - testing local environment"
fi

# Change to project directory
cd "$(dirname "$0")"
echo "📁 Working directory: $(pwd)"

print_status "Testing Python Environment"

# Check Python version
python3 --version
if [ $? -eq 0 ]; then
    echo "[PASS] Python 3 available"
else
    echo "[FAIL] Python 3 not available"
    exit 1
fi

# Check pip
pip3 --version
if [ $? -eq 0 ]; then
    echo "[PASS] pip3 available"
else
    echo "[FAIL] pip3 not available"
fi

print_status "Installing Required Dependencies"

# Install test dependencies
pip3 install pytest psutil || echo "[WARNING] Could not install test dependencies"

# Install core dependencies if needed
pip3 install sentence-transformers faiss-cpu || echo "[WARNING] Could not install ML dependencies"

print_status "Testing File Structure"

# Check critical files exist
critical_files=(
    "backend/hybrid_search.py"
    "backend/search/query_classifier.py"
    "storage/database_setup.py"
    "storage/database_manager.py"
    "knowledge_graph/symbol_collector.py"
    "tests/run_all_tests.py"
)

for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo "[PASS] $file exists"
    else
        echo "[FAIL] $file missing"
        exit 1
    fi
done

print_status "Running Basic Import Tests"

# Test critical imports
python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    # Test Phase 2 imports
    from storage.database_setup import DatabaseSetup
    from storage.database_manager import DatabaseManager
    print('[PASS] Phase 2 storage imports successful')
    
    # Test Phase 3 imports
    from backend.search.query_classifier import QueryClassifier
    from backend.hybrid_search import HybridSearchEngine
    print('[PASS] Phase 3 search imports successful')
    
    print('[PASS] All critical imports working')
except ImportError as e:
    print(f'[FAIL] Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'[FAIL] Error during import: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "[FAIL] Basic import tests failed"
    exit 1
fi

print_status "Running Comprehensive Test Suite"

# Run the comprehensive test suite
python3 tests/run_all_tests.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "[SUCCESS] ALL DOCKER TESTS PASSED!"
    echo "CodeWise Phase 2 and Phase 3 are ready for Docker deployment"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "[FAIL] SOME TESTS FAILED"
    echo "Please review the test output above"
    echo "============================================================"
    exit 1
fi

print_status "Testing in Production-like Environment"

# Test with minimal imports (simulating production constraints)
python3 -c "
import sys
import tempfile
import os
sys.path.insert(0, '.')

print('Testing production-like scenario...')

try:
    # Test database operations with temporary file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db = tmp.name
    
    # Test database initialization
    from storage.database_setup import DatabaseSetup
    db_setup = DatabaseSetup(temp_db)
    success = db_setup.initialize_database()
    
    if success:
        print('[PASS] Database initialization in production mode: OK')
    else:
        print('[FAIL] Database initialization failed')
        sys.exit(1)
    
    # Test query classifier
    from backend.search.query_classifier import QueryClassifier
    classifier = QueryClassifier()
    analysis = classifier.classify_query('test query')
    
    if hasattr(analysis, 'intent') and hasattr(analysis, 'vector_weight'):
        print('[PASS] Query classifier in production mode: OK')
    else:
        print('[FAIL] Query classifier failed')
        sys.exit(1)
    
    # Cleanup
    os.unlink(temp_db)
    
    print('[PASS] Production-like environment test passed')
    
except Exception as e:
    print(f'[FAIL] Production test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "[PASS] Production-like environment test passed"
else
    echo "[FAIL] Production-like environment test failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "[SUCCESS] DOCKER ENVIRONMENT FULLY VALIDATED"
echo "All Phase 2 and Phase 3 components are working correctly"
echo "============================================================"