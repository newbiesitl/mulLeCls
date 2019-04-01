# read data from data set
from utils.data_utils import jigsaw_toxix_ds_get_df
from config import *
df = jigsaw_toxix_ds_get_df()
comments = df["comment_text"].tolist()
comments = [x.replace('\n', ' ') for x in comments]

dump_text = '\n'.join(comments)

# put them into txt
dump_file_name = 'input.txt'
dump_file_path = os.path.join(data_folder, dump_file_name)
with open(dump_file_path, 'w+') as f:
    f.write(dump_text)

# run bert scripts to embed
import subprocess
bashCommend = 'bash gen_embedding.sh'
process = subprocess.Popen(bashCommend.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(output, error)

# extract embeddings from json file and convert into list




# join embeddings with data set dataframe

# extract embedding column and label columns

# convert to X, Y

# fit model and test
