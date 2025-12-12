Final project for Fall 2025 CS410
Credit for inspiration and parts of the codebase to the assignments we covered in class this semester. 
This examine a legal dataset. All zip files should be extracted into the root directory.


Ubuntu enviornnment:
#install minconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh

#define python 3.11 env for the dependecies used
conda create --name cs410 python=3.11
conda activate cs410

# install dependencies
conda install -c conda-forge openjdk=21 maven -y
pip install -r requirements.txt

# download source file from https://storage.courtlistener.com/bulk-data/parentheticals-2025-12-02.csv.bz2 to /data directory
# rename to input.csv
# run clean.py to clean input csv and reduce processing file size to 200,000 rows
# run main.py to generate indexes, queries, and qrels
# insteall ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
# run main2py.py for LLM ollama server 

fin
