import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
os.chdir("..")
from nlp_tools.keyword_tool import get_keywords

if __name__ == "__main__":
    raw_path = os.path.join(current_path, "data/raw_data/ppt/corpus.txt")
    lines = open(raw_path, "r").readlines()
    for line in lines:
        keywords = get_keywords(line, mtd="textrank")
        # print(keywords)
        keywords = [word[0] for word in keywords]
        print(keywords)
