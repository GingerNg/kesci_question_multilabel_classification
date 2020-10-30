import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(current_path)
sys.path.append(current_path)
os.chdir("..")
