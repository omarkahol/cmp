#!/bin/bash

# Script to create a new test file with proper structure
# Usage: ./new_test.sh <test_name>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 my_new_test"
    exit 1
fi

TEST_NAME=$1
TEST_FILE="${TEST_NAME}.cpp"

# Create the basic test file
cat > "$TEST_FILE" << EOF
#include <iostream>
#include <vector>
#include <chrono>
// Add your specific includes here
// #include "your_header.h"

int main() {
    std::cout << "Running ${TEST_NAME} test..." << std::endl;
    
    // Add your test code here
    
    std::cout << "${TEST_NAME} test completed successfully!" << std::endl;
    return 0;
}
EOF

echo "Created new test: $TEST_FILE"
echo ""
echo "The test is automatically available! No Makefile changes needed."
echo ""
echo "To build and run your test:"
echo "  cd tests && make out_${TEST_NAME} && ./out_${TEST_NAME}"
echo ""
echo "Or use uppercase target:"
echo "  cd tests && make $(echo $TEST_NAME | tr '[:lower:]' '[:upper:]')"
echo ""
echo "Or use VS Code tasks:"
echo "  - Open tests/$TEST_FILE"
echo "  - Press F5 or Cmd+F5 to build and run"
echo "  - Use 'Build Current Test' task"