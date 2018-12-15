import sys
import os
import json

""" 
Concatenate generated questions into a singe file 
"""

concatenated = []
direc = "/Users/Akrit/Desktop/premise_emnlp17/qgen/generated_questions"
file_list = os.listdir(direc)
file_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
print file_list
for file in file_list:
    with open(direc + "/" + file, "r") as f:
        questions = json.loads(f.read())
        concatenated.extend(questions)
        f.close()

with open(direc + "/" + "vqa_prem_questions.json", "w") as f:
    json.dump(concatenated, f, indent=4, separators=(',', ': '))
    f.close()