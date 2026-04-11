import traceback
import sys
sys.path.insert(0, '.')
try:
    import inference
    print('imported!')
except Exception as e:
    traceback.print_exc()
