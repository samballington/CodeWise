#!/usr/bin/env python3
"""
Comprehensive test runner for Response Pipeline Consolidation Fix
Runs all test suites and provides a summary
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from test_response_consolidator_fix import TestResponseConsolidatorFix
from test_pipeline_selection import TestPipelineSelection
from test_response_pipeline_integration import TestResponsePipelineIntegration
from test_response_edge_cases import TestResponseEdgeCases


async def run_all_tests():
    """Run all test suites and provide comprehensive results"""
    
    print("🚀 RESPONSE PIPELINE CONSOLIDATION FIX - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    # Test Suite 1: Response Consolidator Fix
    print("\n📦 PART 1: Response Consolidator Logic Tests")
    print("-" * 50)
    
    consolidator_tests = TestResponseConsolidatorFix()
    
    test_methods = [
        ("test_structured_response_no_content_leakage", consolidator_tests.test_structured_response_no_content_leakage),
        ("test_markdown_response_uses_output_field", consolidator_tests.test_markdown_response_uses_output_field),
        ("test_malformed_structured_response_fallback", consolidator_tests.test_malformed_structured_response_fallback),
        ("test_empty_response_data_handling", consolidator_tests.test_empty_response_data_handling),
        ("test_mixed_response_sources_priority", consolidator_tests.test_mixed_response_sources_priority)
    ]
    
    for test_name, test_method in test_methods:
        total_tests += 1
        try:
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            print(f"✅ {test_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed_tests += 1
    
    # Test Suite 2: Pipeline Selection
    print("\n🔀 PART 2: Pipeline Selection Logic Tests")
    print("-" * 50)
    
    pipeline_tests = TestPipelineSelection()
    
    test_methods = [
        ("test_structured_response_uses_json_pipeline_only", pipeline_tests.test_structured_response_uses_json_pipeline_only),
        ("test_markdown_response_uses_markdown_pipeline_only", pipeline_tests.test_markdown_response_uses_markdown_pipeline_only),
        ("test_json_parsing_detection", pipeline_tests.test_json_parsing_detection),
        ("test_pipeline_selection_logic", pipeline_tests.test_pipeline_selection_logic)
    ]
    
    for test_name, test_method in test_methods:
        total_tests += 1
        try:
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            print(f"✅ {test_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed_tests += 1
    
    # Test Suite 3: Integration Tests
    print("\n🔗 PART 3: Integration Tests")
    print("-" * 50)
    
    integration_tests = TestResponsePipelineIntegration()
    
    test_methods = [
        ("test_end_to_end_structured_response", integration_tests.test_end_to_end_structured_response),
        ("test_end_to_end_markdown_response", integration_tests.test_end_to_end_markdown_response),
        ("test_mixed_response_sources", integration_tests.test_mixed_response_sources),
        ("test_malformed_structured_response", integration_tests.test_malformed_structured_response),
        ("test_empty_response_data", integration_tests.test_empty_response_data),
        ("test_mermaid_validation_integration", integration_tests.test_mermaid_validation_integration)
    ]
    
    for test_name, test_method in test_methods:
        total_tests += 1
        try:
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            print(f"✅ {test_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed_tests += 1
    
    # Test Suite 4: Edge Cases
    print("\n⚠️ PART 4: Edge Case Tests")
    print("-" * 50)
    
    edge_case_tests = TestResponseEdgeCases()
    
    test_methods = [
        ("test_malformed_structured_response", edge_case_tests.test_malformed_structured_response),
        ("test_empty_response_data", edge_case_tests.test_empty_response_data),
        ("test_multiple_structured_responses", edge_case_tests.test_multiple_structured_responses),
        ("test_structured_response_with_empty_sections", edge_case_tests.test_structured_response_with_empty_sections),
        ("test_raw_response_fallback", edge_case_tests.test_raw_response_fallback),
        ("test_synthesis_response_priority", edge_case_tests.test_synthesis_response_priority),
        ("test_max_iterations_response", edge_case_tests.test_max_iterations_response),
        ("test_error_aggregation", edge_case_tests.test_error_aggregation)
    ]
    
    for test_name, test_method in test_methods:
        total_tests += 1
        try:
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            print(f"✅ {test_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed_tests += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Response Pipeline Consolidation Fix is working correctly.")
        print("\n✅ IMPLEMENTATION STATUS: READY FOR DEPLOYMENT")
        print("\nKey Achievements:")
        print("• ✅ Eliminated content duplication")
        print("• ✅ Fixed pipeline selection logic")
        print("• ✅ Proper structured response handling")
        print("• ✅ Clean separation between JSON and markdown pipelines")
        print("• ✅ Comprehensive error handling")
        print("• ✅ Mermaid validation integration")
    else:
        print(f"\n⚠️ {failed_tests} TESTS FAILED - Review and fix issues before deployment")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)