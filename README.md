# AIR-retriever
AIR retriever for Multi-Hop QA - ACL 2020 [paper] (https://arxiv.org/abs/2005.01218)

## Running Experiments:

1] Please download the MultiRC dataset from https://github.com/CogComp/multirc
The train and dev sets are available in the above link.

2] Running "python3 AIR_evidence_retrieval_scores.py" shows the justification selection performance of AIR. Please change the directory of GLoVe embeddings (line 16 of the file) as per your GLoVe files. 

3] Running "python3 main_MultiRC_passages_from_topN_Iterative_alignments_PARALLEL_evidences.py" generates the train and dev files for the QA tasks with various parallel evidences. We had followed the binary classification approach for every candidate answer in the QA task (https://arxiv.org/abs/2005.01218), hence the files are in MRPC format which makes it easier to train any transformer based approach (RoBERTa, XLnet or BERT) from the huggingface library. 

QA files in MRPC format can be downloaded from [here](https://drive.google.com/file/d/1hyMGTKCu_4LZir9VPPnsNqh05gYMD0aN/view?usp=sharing)

