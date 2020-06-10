import json
import os
from get_subgraph import get_iterative_alignment_justifications_non_parameteric, get_iterative_alignment_justifications_non_parameteric_withQreform_flag, get_iterative_alignment_justifications_unsupervised_semantic_drift, get_iterative_alignment_justifications_non_parameteric_LEXICAL
from Compute_F1 import F1_Score_just_quality
import math
import numpy as np
# from get_regression_labels import get_regression_predictions
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

POCC_subgraph_size = 5  ## for POCC
output_file_dir = "MultiRC_BM25_vs_POCC_justification_quality_score/"

if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)

split_set = "dev"

# perfectly_scored_ginds = [15, 16, 66, 101, 103, 105, 137, 142, 177, 209, 210, 211, 219, 220, 227, 233, 234, 235, 256, 257, 264, 278, 287, 321, 322, 342, 373, 377, 430, 443, 444, 445, 450, 454, 460, 461, 462, 475, 476, 534, 535, 593, 632, 657, 658, 689, 693, 694, 695, 704, 712, 737, 751, 893, 897, 923, 924, 929, 936, 958, 968, 996, 997, 1000, 1013, 1030, 1036, 1064, 1065, 1095, 1096, 1112, 1120, 1121, 1124, 1138, 1140, 1153, 1158, 1160, 1297, 1347, 1348, 1350, 1362, 1363, 1371, 1372, 1385, 1387, 1449, 1465, 1468, 1469, 1470, 1508, 1514, 1515, 1516, 1564, 1566, 1567, 1635, 1638, 1639, 1641, 1642, 1656, 1686, 1722, 1741, 1745, 1751, 1779, 1787, 1791, 1802, 1803, 1862, 1909, 1910, 1911, 1917, 1937, 1938, 1939, 1940, 1946, 1947, 1948, 1951, 1954, 1955, 1968, 1971, 1975, 1988, 2003, 2054, 2060, 2064, 2065, 2066] ## look for a short passage example amongst these passages.
perfectly_scored_ginds = [264]

if split_set == "dev":
   input_file_name =  "dev_83-fixedIds.json" # "dev_after_coref_resolved_only_pronouns_replaced.json"  #
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

Proportion_perfect_justification_set = []
Coherence_questions = []

ROCC_ranked_recall_coverage = []
ROCC_ranked_precision_coverage = []


Write_file = open(output_file_dir + out_file_name, "w")
with open(input_file_name) as json_file:
    json_data = json.load(json_file)
    Gold_sentences_IDs = []
    All_gold_sentences = []
    All_gold_sentences_corresponding_ques = []
    All_gold_sentences_query_reform_scores = []
    All_gold_sentences_reformed_queries = []


    Predicted_sent_IDs = []
    Predicted_sent_POCC_IDs = []

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

        # print (len(para_ques['paragraph']["questions"]),para_ques['id'])
        # print (para_ques['paragraph']["questions"][1],para_ques['id'])

        for qind, ques_ans1 in enumerate(para_ques['paragraph']["questions"]):
            question_text = ques_ans1['question']

            for cand_ind, cand_ans in enumerate(ques_ans1['answers']):

                if cand_ans['isAnswer'] == True:
                    Gold_sentences_IDs.append(ques_ans1["sentences_used"])
                    All_gold_sentences.append([current_KB_passage_sents[int(i1)] for i1 in ques_ans1["sentences_used"]])
                    All_gold_sentences_corresponding_ques.append(question_text+ " || " + cand_ans["text"])
                    answer_text1 = cand_ans["text"]
                    pred_sent_indexes_POCC, ROCC_ranked_PRF=get_iterative_alignment_justifications_non_parameteric(question_text, answer_text1, current_KB_passage_sents, MultiRC_idf_vals, embeddings_index, ques_ans1["sentences_used"], emb_size=emb_size, subgraph_size=POCC_subgraph_size)  ## Alignment over embeddings for sentence selection
                    # # pred_sent_indexes_POCC, ROCC_ranked_PRF, query_reform_score, all_reformed_queries=get_iterative_alignment_justifications_non_parameteric_withQreform_flag(question_text, answer_text1, current_KB_passage_sents, MultiRC_idf_vals, embeddings_index, ques_ans1["sentences_used"], emb_size=emb_size, subgraph_size=POCC_subgraph_size)  ## Alignment over embeddings for sentence selection
                    # All_gold_sentences_query_reform_scores.append(query_reform_score)
                    # All_gold_sentences_reformed_queries.append(all_reformed_queries)
                   #  (ques_text, answer_text, justifications, IDF_vals, embedding_index, emb_size = 100, subgraph_size = 5, return_ROCC_vals = 0)
                    Predicted_sent_POCC_IDs.append(pred_sent_indexes_POCC)
                    ROCC_ranked_recall_coverage.append(ROCC_ranked_PRF[1])
                    ROCC_ranked_precision_coverage.append(ROCC_ranked_PRF[0])

P_R_F1score, perfectly_retrieved_justifications = F1_Score_just_quality(Gold_sentences_IDs, Predicted_sent_POCC_IDs)
print ("precision, recall and Fscores are: ", P_R_F1score, perfectly_retrieved_justifications)


# print ("The ranked recall of ROCC is: ", sum(ROCC_ranked_recall_coverage)/float(len(ROCC_ranked_recall_coverage)), ROCC_ranked_recall_coverage )
# print ("The ranked precision of ROCC is: ", sum(ROCC_ranked_precision_coverage)/float(len(ROCC_ranked_precision_coverage)), ROCC_ranked_precision_coverage )






