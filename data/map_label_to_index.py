import os
import json

label_set = set()
for f in os.listdir('/mnt/petrelfs/share_data/wuchaoyi/SAM/Knowledge_Data/MedCPT_Query_Encoder/text_prompts'): # mod_lab.npy / none.npy / lab.npy
    name = f[:-4]
    label_set.add(name.lower())

dict = {'none':0}
index = 1
for name in label_set:
    dict[name] = index
    index += 1

with open('/mnt/petrelfs/share_data/wuchaoyi/SAM/Knowledge_Data/label2index.json', 'w') as f:
    json.dump(dict, f, indent=4)
    