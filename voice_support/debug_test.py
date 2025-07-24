import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import main
    print("Successfully imported main module")
    print(f"qa_chain type: {type(main.qa_chain)}")
    print(f"qa_chain methods: {[method for method in dir(main.qa_chain) if not method.startswith('_')]}")
    
    # Test the qa_chain directly
    print("\nTesting qa_chain directly...")
    result = main.qa_chain({"query": "测试问题"})
    print(f"Direct call result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
