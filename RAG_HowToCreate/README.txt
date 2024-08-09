Commands to create the environment 'RAG':
1. Create the conda environment:
   conda create -n RAG python=3.8
2. Activate the conda environment:
   conda activate RAG
3. Install conda packages:
   conda env update --file RAG_environment.yml --prune
4. Install pip packages:
   pip install -r RAG_requirements.txt
