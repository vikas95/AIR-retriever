import json
import os
from get_subgraph import get_iterative_alignment_justifications_non_parameteric, get_iterative_alignment_justifications_unsupervised_semantic_drift, get_iterative_alignment_justifications_non_parameteric_LEXICAL, get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence

from Compute_F1 import F1_Score_just_quality, Rouge_n_scores
import math
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

##### alignment part here :

embeddings_index = {}

f = open(os.path.join("/Users/vikasy/Glove_vectors/","glove.6B.100d.txt"),'r', encoding='utf-8')
# f = open(os.path.join("/Users/vikasy/Glove_vectors/","glove.840B.300d.txt"),'r', encoding='utf-8')
# f = open("all_emb.txt",'r', encoding='utf-8')
# f = open("GW_vectors.txt", 'r', encoding='utf-8')  ## gives a lot lesser performance.

#f = open('ss_qz_04.dim50vecs.txt')
for line in f:
    values = line.split()
    word = lmtzr.lemmatize(values[0].lower())
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       b = np.linalg.norm(coefs, ord=2)
       coefs = coefs / float(b)
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
print("Embedding size is: ", emb_size)

POCC_subgraph_size = 5 ## for POCC
Num_Parallel_evidences = 5

split_set = "dev" ## "train"  ## change this value to generate either training file or dev file for the QA task.

output_file_dir = "MultiRC_justifications_Iterative_Alignment_CLASSIFICATION_parallel_concat_evidence_"+str(Num_Parallel_evidences) # + str(POCC_subgraph_size) +"_CLASSIFICATION/"
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)

if split_set == "dev":
   input_file_name =  "dev_83-fixedIds.json"
   out_file_name = "dev.tsv"
   test_out_file_name = "test.tsv"
   Test_Write_file = open(output_file_dir + test_out_file_name, "w")
elif split_set == "train":
   input_file_name = "train_456-fixedIds.json"
   out_file_name = "train.tsv"


with open("MultiRC_IDF_vals.json") as json_file:
    MultiRC_idf_vals = json.load(json_file)

total_ques = 0
All_KB_passages = []

# output_file_dir = "MultiRC_BM25_passage_"+str(num_sent)+"/"
# output_file_dir = "MultiRC_POCC_passage_"+str(num_sent)+"SENT/"
# output_file_dir = "MultiRC_POCC_passage_Final_m_SENT_2_3_4_5_6_withIDF/"
Write_file = open(output_file_dir + out_file_name, "w")


if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
# first_line = "index	genre	filename	year	old_index	source1	source2	sentence1	sentence2	score"
first_line = "dummy"+ "\t" + "dummy" + "\t" + "dummy" + "_" + "dummy" + "\t" + "dummy" + " " + "dummy" + "\t" + "dummy"

Write_file.write(first_line+"\n")

if split_set == "dev":
   Test_Write_file.write(first_line+"\n")

number_of_lines = 0
with open(input_file_name) as json_file:
    json_data = json.load(json_file)

    for para_ques in json_data["data"]:
        print ("we are at this question: ", total_ques)
        current_KB_passage_sents = []
        total_ques += len(para_ques['paragraph']["questions"])
        num_of_justifications = para_ques['paragraph']["text"].count("<br>")
        # print (num_of_justifications, para_ques['paragraph']["text"])
        for i in range(num_of_justifications):
            start_index = para_ques['paragraph']["text"].find("<b>Sent "+str(i+1)+ ": </b>") + len("<b>Sent "+str(i+1)+ ": </b>")
            end_index = para_ques['paragraph']["text"].find("<b>Sent "+str(i+2)+ ": </b>")
            if i == num_of_justifications-1:
                current_KB_passage_sents.append(para_ques['paragraph']["text"][start_index:end_index].replace("<br", ""))
            else:
                current_KB_passage_sents.append(para_ques['paragraph']["text"][start_index:end_index].replace("<br>", ""))

        All_KB_passages.append(current_KB_passage_sents)
        for qind, ques_ans1 in enumerate(para_ques['paragraph']["questions"]):
            question_text = ques_ans1['question']

            GOLD_passage_IDS = ques_ans1["sentences_used"]
            GOLD_passage_IDS_str = [str(ind1) for ind1 in ques_ans1["sentences_used"]]

            for cand_ind, cand_ans in enumerate(ques_ans1['answers']):
                # if split_set == "test":
                #    cand_ans.update({'isAnswer':True})
                #    ques_ans1.update({"sentences_used":[1,2]})


                # pred_sent_indexes_IA, ROCC_ranked_PRF = get_iterative_alignment_justifications_non_parameteric(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, embeddings_index,ques_ans1["sentences_used"], emb_size=emb_size, subgraph_size=POCC_subgraph_size)  ## Alignment over embeddings for sentence selection
                pred_sent_indexes_IA = get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, embeddings_index,ques_ans1["sentences_used"], Num_Parallel_evidences, emb_size=emb_size, subgraph_size=POCC_subgraph_size)  ## Alignment over embeddings for sentence selection
                # pred_sent_indexes_IA = get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, embeddings_index,ques_ans1["sentences_used"], Num_Parallel_evidences, emb_size=emb_size, subgraph_size=POCC_subgraph_size)  ## Alignment over embeddings for sentence selection

                iterative_alignment_passage = " ".join( [current_KB_passage_sents[i1] for i1 in pred_sent_indexes_IA] )

                if cand_ans['isAnswer'] == True:
                   new_line = str(1) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + iterative_alignment_passage + "\n"

                else:
                   new_line = str(0) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + iterative_alignment_passage + "\n"

                number_of_lines += 1

                Write_file.write(new_line)
                # Write_file.write(new_line_2)

                if split_set == "dev":
                    Test_Write_file.write(new_line)
                    # Test_Write_file.write(new_line_2)







