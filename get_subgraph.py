from Preprocess_datasets import Preprocess_QA_sentences, Preprocess_QA_sentences_Quoref
from Graph_nodes import get_all_combination_withCoverage_best_graph, get_all_combination_withCoverage_best_graph_Cand_boost, get_all_combination_withCoverage_best_graph_Cand_boost_withIDF, get_all_combination_withCoverage_best_graph_Cand_boost_ALL, \
                        get_all_combination_withCoverage_Alignment_IDF, get_all_combination_withCoverage_SOFT_Alignment_IDF, get_all_combination_withCoverage_Alignment_Regression
from BM25_function import get_BM25_scores
# from Alignment_function import get_alignment_score
import numpy as np
from collections import Counter
from Overlap_analysis import calculate_overlap, calculate_all_overlap, calculate_overlap_labels, get_union, get_intersection, calculate_kappa, get_normalized_scores
from itertools import combinations
import math
from Compute_F1 import F1_Score_individual_just

"""

Put the condition of decent sized justification also in these functions
"""
def get_query_term_coverage(ques_text, justification_chain):
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    just_terms = []
    for just1 in justification_chain:
        just_terms += Preprocess_QA_sentences(just1,1)

    return len(set(ques_terms).intersection(set(just_terms)))/float(len(set(ques_terms)))

def sent_Emb(ques1, embeddings_index, emb_size, IDF_val, ques_text = 0, object_subject_list=[]):
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = []  ##### IDF is of size = 1 coz its a value
    tokens_not_found_embeddings = []
    tokens_embeddings_found = []
    for q_term in ques1:
        if q_term in embeddings_index:
           Ques_Matrix = np.append(Ques_Matrix, np.array([np.asarray(embeddings_index[q_term])]), axis=0)
           tokens_embeddings_found.append(q_term)
        else:
           tokens_not_found_embeddings.append(q_term)

        if ques_text == 1:
            if q_term in object_subject_list:
                important_term_coefficient = 1
            else:
                important_term_coefficient = 1

            if q_term in IDF_val:
                IDF_Mat.append(important_term_coefficient*IDF_val[q_term])
            else:
                IDF_Mat.append(important_term_coefficient*3)
                # print ("the unknown IDF term is: ", q_term)
    if ques_text == 1:
       return Ques_Matrix, IDF_Mat, tokens_not_found_embeddings, tokens_embeddings_found
    else:
       return Ques_Matrix, tokens_not_found_embeddings, tokens_embeddings_found


def compute_alignment_vector(ques_matrix, ques_toks_nf, ques_toks_found, just_sent_matrix, just_toks_nf, threshold = 0.95):
    just_sent_matrix = just_sent_matrix.transpose()
    Score = np.matmul(ques_matrix, just_sent_matrix)
    Score = np.sort(Score, axis=1)
    max_score1 = Score[:, -1:]  ## taking the highest element column
    max_score1 = np.asarray(max_score1).flatten()

    max_score1 = [1 if s1>=threshold else 0 for s1 in max_score1]
    # max_score1 = [s1 for s1 in max_score1]

    remaining_terms = []

    for i1, s1 in enumerate(max_score1):
        if s1 == 0:
           remaining_terms.append(ques_toks_found[i1])

    for t1 in ques_toks_nf:
        if t1 in just_toks_nf:
           max_score1.append(1)
        else:
           max_score1.append(0)
           remaining_terms.append(t1)

    return remaining_terms

def compute_alignment_score(ques_matrix, ques_toks_nf, ques_idf_vector, just_sent_matrix, just_toks_nf, IDF_vals):
    just_sent_matrix = just_sent_matrix.transpose()
    Score = np.matmul(ques_matrix, just_sent_matrix)
    Score = np.sort(Score, axis=1)
    max_score1 = Score[:, -1:]  ## taking 3 highest element columns
    max_score1 = np.asarray(max_score1).flatten()

    final_just_score = [a1*b1 for a1,b1 in zip(max_score1, ques_idf_vector)]

    for t1 in ques_toks_nf:
        if t1 in just_toks_nf:
            if t1 in IDF_vals:
               final_just_score.append(IDF_vals[t1])
            else:
               final_just_score.append(3) ## ~3 was the average IDF score

    return sum(final_just_score)

def get_naive_alignment_justification(ques_text, answer_text, justifications, IDF_vals, embedding_index, emb_size, subgraph_size=2):
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)

    ques_ans_matrix, ques_ans_IDF_mat, ques_ans_toks_nf, ques_ans_toks_found = sent_Emb(ques_terms + answer_terms, embedding_index, emb_size, IDF_vals, 1)

    Justification_ques_ans_remaining_terms = {}

    Final_alignment_scores = []
    num_remaining_terms = []
    for jind1, just1 in enumerate(justifications):
        just_terms = Preprocess_QA_sentences(just1, 1)
        # All_justification_terms.update({jind1: just_terms})
        just_matrix, just_toks_nf, just_toks_found = sent_Emb(just_terms, embedding_index, emb_size, IDF_vals)
        Justification_ques_ans_remaining_terms.update({jind1: compute_alignment_vector(ques_ans_matrix, ques_ans_toks_nf, ques_ans_toks_found, just_matrix, just_toks_nf)})

        jind_score = compute_alignment_score(ques_ans_matrix, ques_ans_toks_nf, ques_ans_IDF_mat, just_matrix, just_toks_nf, IDF_vals)
        # BM25_scores.update({jind1: jind_score})
        num_remaining_terms.append(len(Justification_ques_ans_remaining_terms[jind1]))
        Final_alignment_scores.append(jind_score)

    Final_index = list(np.argsort(Final_alignment_scores)[::-1])[:subgraph_size]
    # Final_index = list(np.argsort(num_remaining_terms)[0:10])

    return Final_index


def get_naive_alignment_justification(ques_text, answer_text, justifications, IDF_vals, embedding_index, emb_size=100, subgraph_size=2):
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)

    ques_ans_matrix, ques_ans_IDF_mat, ques_ans_toks_nf, ques_ans_toks_found = sent_Emb(ques_terms + answer_terms, embedding_index, emb_size, IDF_vals, 1)

    Justification_ques_ans_remaining_terms = {}

    Final_alignment_scores = []
    num_remaining_terms = []
    for jind1, just1 in enumerate(justifications):
        just_terms = Preprocess_QA_sentences(just1, 1)
        # All_justification_terms.update({jind1: just_terms})
        just_matrix, just_toks_nf, just_toks_found = sent_Emb(just_terms, embedding_index, emb_size, IDF_vals)
        Justification_ques_ans_remaining_terms.update({jind1: compute_alignment_vector(ques_ans_matrix, ques_ans_toks_nf, ques_ans_toks_found, just_matrix, just_toks_nf)})

        jind_score = compute_alignment_score(ques_ans_matrix, ques_ans_toks_nf, ques_ans_IDF_mat, just_matrix, just_toks_nf, IDF_vals)
        # BM25_scores.update({jind1: jind_score})
        num_remaining_terms.append(len(Justification_ques_ans_remaining_terms[jind1]))
        Final_alignment_scores.append(jind_score)

    Final_index = list(np.argsort(Final_alignment_scores)[::-1])# [:subgraph_size]
    # Final_index = list(np.argsort(num_remaining_terms)[0:10])
    return Final_index

def get_alignment_justification(ques_terms, answer_terms, justifications, embedding_index, emb_size, IDF_vals):

    ques_ans_matrix, ques_ans_IDF_mat, ques_ans_toks_nf, ques_ans_toks_found = sent_Emb(ques_terms + answer_terms, embedding_index, emb_size, IDF_vals, 1)

    Justification_ques_ans_remaining_terms = {}

    Final_alignment_scores = []
    num_remaining_terms = []
    for jind1, just1 in enumerate(justifications):
        just_terms = Preprocess_QA_sentences(just1, 1)
        # All_justification_terms.update({jind1: just_terms})
        just_matrix, just_toks_nf, just_toks_found = sent_Emb(just_terms, embedding_index, emb_size, IDF_vals)
        Justification_ques_ans_remaining_terms.update({jind1: compute_alignment_vector(ques_ans_matrix, ques_ans_toks_nf, ques_ans_toks_found, just_matrix, just_toks_nf)})

        jind_score = compute_alignment_score(ques_ans_matrix, ques_ans_toks_nf, ques_ans_IDF_mat, just_matrix, just_toks_nf, IDF_vals)
        # BM25_scores.update({jind1: jind_score})
        num_remaining_terms.append(len(Justification_ques_ans_remaining_terms[jind1]))
        Final_alignment_scores.append(jind_score)

    Final_index = list(np.argsort(Final_alignment_scores)[::-1]) ## the higher alignment score, the more similar it is
    # Final_index = list(np.argsort(num_remaining_terms)) ## the less remaining terms, the more similar it is

    return Final_index, Justification_ques_ans_remaining_terms[Final_index[0]], Justification_ques_ans_remaining_terms

def get_LEXICAL_justification(ques_terms, answer_terms, justifications):

    All_BM25_scores = get_BM25_scores(justifications, " ".join(ques_terms+answer_terms))
    Justification_ques_ans_remaining_terms = {}

    query_terms = set(ques_terms+answer_terms)

    num_remaining_terms = []
    for jind1, just1 in enumerate(justifications):
        just_terms = Preprocess_QA_sentences(just1, 1)
        remainning_terms_current_justification = list(query_terms - set(just_terms))
        Justification_ques_ans_remaining_terms.update({jind1:remainning_terms_current_justification})

        num_remaining_terms.append(len(Justification_ques_ans_remaining_terms[jind1]))

    Final_index = list(np.argsort(All_BM25_scores)[::-1])
    # Final_index = list(np.argsort(num_remaining_terms)[0:10])

    return Final_index, Justification_ques_ans_remaining_terms[Final_index[0]], Justification_ques_ans_remaining_terms


def One_iteration_block(Final_indexes, first_iteration_index1, remaining_toks1_3, ques_terms, answer_terms, justifications, embedding_index, emb_size, IDF_vals):

    try:
        selected_just_toks = Preprocess_QA_sentences(justifications[Final_indexes[-1]], 1)
        # try:
        #     selected_just_toks += Preprocess_QA_sentences(justifications[Final_indexes[-2]], 1)
        #     selected_just_toks += Preprocess_QA_sentences(justifications[Final_indexes[-3]], 1)
        # except IndexError:
        #     pass

    except IndexError:
        print ("the error is coming because ", Final_indexes, len(justifications)) ## please pardon these error messages, they do not appear when running the main file and were used only for debugging
        selected_just_toks = Preprocess_QA_sentences(justifications[Final_indexes[0]], 1)
    if len(remaining_toks1_3) <= 1:  ## which can be considered as a very short query
        new_query_terms = remaining_toks1_3  + list(set(selected_just_toks) - set(ques_terms + answer_terms))

    else:
        new_query_terms = remaining_toks1_3
        # new_query_terms = remaining_toks1_3 + list(set(selected_just_toks))
    second_iteration_index1, remaining_toks1_4, remaining_toks2_All = get_alignment_justification(new_query_terms, [], justifications, embedding_index, emb_size, IDF_vals)

    for i1 in second_iteration_index1:
        if i1 in Final_indexes:
            pass
        else:
            if len(set(remaining_toks1_3).intersection(set(remaining_toks2_All[i1]))) == len(set(remaining_toks1_3)):  ### i.e. none of the previously remaining ques+ans terms were covered in this iteration
                return Final_indexes, [], []
            Final_indexes.append(i1)
            remaining_toks1_4 = remaining_toks2_All[i1]
            break

    return Final_indexes, second_iteration_index1, remaining_toks1_4

def One_iteration_block_LEXICAL(Final_indexes, first_iteration_index1, remaining_toks1_3, ques_terms, answer_terms, justifications):

    try:
        selected_just_toks = Preprocess_QA_sentences(justifications[Final_indexes[-1]], 1)
    except IndexError:
        print ("the error is coming because ", Final_indexes, len(justifications))
        selected_just_toks = Preprocess_QA_sentences(justifications[Final_indexes[0]], 1)

    if len(remaining_toks1_3) <= 1:  ## which can be considered as a very short query
        new_query_terms = remaining_toks1_3 + list(set(selected_just_toks) - set(ques_terms + answer_terms))
        # print ("yes, we do have some really short queries. ")
    else:
        new_query_terms = remaining_toks1_3
    # justifications[i1] = ""  ## the justifications that are already added, remove them from the list.
    second_iteration_index1, remaining_toks1_4, remaining_toks2_All = get_LEXICAL_justification(new_query_terms, [], justifications)

    for i1 in second_iteration_index1:
        if i1 in Final_indexes:
            pass
        else:
            if len(set(remaining_toks1_3).intersection(set(remaining_toks2_All[i1]))) == len(set(remaining_toks1_3)):  ### i.e. none of the previously remaining ques+ans terms were covered in this iteration
                return Final_indexes, [], []
            Final_indexes.append(i1)
            remaining_toks1_4 = remaining_toks2_All[i1]
            break

    return Final_indexes, second_iteration_index1, remaining_toks1_4

def One_iteration_block_LEXICAL_Semantic_Drift(Final_indexes, first_iteration_index1, remaining_toks1_3, ques_terms, answer_terms, justifications):
    selected_just_toks = Preprocess_QA_sentences(justifications[first_iteration_index1[0]], 1)

    new_query_terms = remaining_toks1_3 + selected_just_toks

    second_iteration_index1, remaining_toks1_4, remaining_toks2_4 = get_LEXICAL_justification(new_query_terms, [], justifications)

    for i1 in second_iteration_index1:
        if i1 in Final_indexes:
            pass
        else:
            Final_indexes.append(i1)
            break

    return Final_indexes, second_iteration_index1, remaining_toks1_4

##### the iterative algorithm parameter free

def get_iterative_alignment_justifications_non_parameteric(ques_text, answer_text, justifications, IDF_vals, embedding_index, gold_justification_indexes, emb_size = 100, subgraph_size = 5):  ## Alignment over embeddings for sentence selection
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)
    Final_indexes = []
    ###### First iteration is here
    first_iteration_index1, remaining_toks1, remaining_toks2 = get_alignment_justification(ques_terms + answer_terms, [], justifications, embedding_index, emb_size, IDF_vals)

    Final_indexes += [first_iteration_index1[0]]  # , first_iteration_index1[1]]

    for i in range(6):  ## i.e. we are making 6 iteration loop to keep the experiments fast but even if you make it 100 or the same as number of sentences in the paragraph, the F1 score remains the same as iteration is completing within first 3-4 loop

        Final_indexes, first_iteration_index1, remaining_toks1 = One_iteration_block(Final_indexes, first_iteration_index1, remaining_toks1, ques_terms, answer_terms, justifications, embedding_index, emb_size, IDF_vals)  ## second iteration

        if len(first_iteration_index1) == 0:
            ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
            return Final_indexes, ROCC_ranked_PRF

    ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
    # print ("the final indices look like ", Final_indexes)
    return Final_indexes, ROCC_ranked_PRF

def get_iterative_alignment_justifications_non_parameteric_withQreform_flag(ques_text, answer_text, justifications, IDF_vals, embedding_index, gold_justification_indexes, emb_size = 100, subgraph_size = 5):  ## Alignment over embeddings for sentence selection
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)
    Final_indexes = []
    ###### First iteration is here
    All_Reformed_queries = [[ques_terms+answer_terms]]
    first_iteration_index1, remaining_toks1, remaining_toks2 = get_alignment_justification(ques_terms + answer_terms, [], justifications, embedding_index, emb_size, IDF_vals)

    Final_indexes += [first_iteration_index1[0]]  # , first_iteration_index1[1]]
    overall_query_reform_score = 0

    for i in range(6):  ## i.e. we are making 6 iteration loop to keep the experiments fast but even if you make it 100 or the same as number of sentences in the paragraph, the F1 score remains the same as iteration is completing within first 3-4 loop

        Final_indexes, first_iteration_index1, remaining_toks1, query_reform_flag, reformed_query = One_iteration_block(Final_indexes, first_iteration_index1, remaining_toks1, ques_terms, answer_terms, justifications, embedding_index, emb_size, IDF_vals)  ## second iteration
        overall_query_reform_score += query_reform_flag
        All_Reformed_queries.append(reformed_query)
        if len(first_iteration_index1) == 0:
            ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
            return Final_indexes, ROCC_ranked_PRF, overall_query_reform_score, All_Reformed_queries

    ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
    # print ("the final indices look like ", Final_indexes)
    return Final_indexes, ROCC_ranked_PRF, overall_query_reform_score, All_Reformed_queries


def get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(ques_text, answer_text, justifications, IDF_vals, embedding_index, gold_justification_indexes, parallel_evidence_num=3, emb_size = 100, subgraph_size = 5):  ## Alignment over embeddings for sentence selection
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)

    All_final_indexes = []
    for num_chain in range(parallel_evidence_num): ### creating a parallel evidence chain of size n
        Final_indexes = []
        ###### First iteration is here
        first_iteration_index1, remaining_toks1, remaining_toks2 = get_alignment_justification(ques_terms + answer_terms, [], justifications, embedding_index, emb_size, IDF_vals)
        for top_ind in first_iteration_index1:
            if top_ind in All_final_indexes:
               pass
            else:
                Final_indexes += [top_ind]  # , first_iteration_index1[1]]
                break
        if len(Final_indexes) == 0:
           Final_indexes += [first_iteration_index1[0]]
           print ("so we did come in this case ")
        for i in range(6):  ## i.e. we are making 6 iteration loop to keep the experiments fast but even if you make it 100 or the same as number of sentences in the paragraph, the F1 score remains the same as iteration is completing within first 3-4 loop

            Final_indexes, first_iteration_index1, remaining_toks1 = One_iteration_block(Final_indexes, first_iteration_index1, remaining_toks1, ques_terms, answer_terms, justifications, embedding_index, emb_size, IDF_vals)  ## second iteration

            if len(first_iteration_index1) == 0:
                if num_chain == parallel_evidence_num-1:
                   All_final_indexes += Final_indexes
                   All_final_indexes = list(set(All_final_indexes))
                   return All_final_indexes
                else:
                    ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
                    break

        All_final_indexes += Final_indexes
    All_final_indexes = list(set(All_final_indexes))
    return All_final_indexes

def get_iterative_alignment_justifications_non_parameteric_LEXICAL(ques_text, answer_text, justifications, IDF_vals, embedding_index, gold_justification_indexes, emb_size = 100, subgraph_size = 5):  ## Alignment over embeddings for sentence selection
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)
    Final_indexes = []
    ###### First iteration is here
    first_iteration_index1, remaining_toks1, remaining_toks2 = get_LEXICAL_justification(ques_terms+answer_terms, [], justifications)

    Final_indexes+=[first_iteration_index1[0]] # , first_iteration_index1[1]]

    for i in range(6): ## i.e. we are making 10 iteration loop to keep the experiments fast but even if you make it 100 or the same as number of sentences in the paragraph, the F1 score remains the same as iteration is completing within first 3-4 loop

        Final_indexes, first_iteration_index1, remaining_toks1 = One_iteration_block_LEXICAL(Final_indexes, first_iteration_index1, remaining_toks1, ques_terms, answer_terms, justifications)  ## second iteration

        # """ ## always comment these whenever conducting experiments of semantic drift.
        if len(first_iteration_index1) == 0:
           ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
           return Final_indexes, ROCC_ranked_PRF
        # """

    ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
    # print ("the final indices look like ", Final_indexes)
    return Final_indexes, ROCC_ranked_PRF

def get_iterative_alignment_justifications_non_parameteric_LEXICAL_semantic_drift(ques_text, answer_text, justifications, IDF_vals, embedding_index, gold_justification_indexes, emb_size = 100, subgraph_size = 5):  ## Alignment over embeddings for sentence selection
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)
    Final_indexes = []
    ###### First iteration is here
    first_iteration_index1, remaining_toks1, remaining_toks2 = get_LEXICAL_justification(ques_terms+answer_terms, [], justifications)

    Final_indexes+=[first_iteration_index1[0]] # , first_iteration_index1[1]]

    for i in range(4): ## i.e. we are making 10 iteration loop to keep the experiments fast but even if you make it 100 or the same as number of sentences in the paragraph, the F1 score remains the same as iteration is completing within first 3-4 loop

        Final_indexes, first_iteration_index1, remaining_toks1 = One_iteration_block_LEXICAL_Semantic_Drift(Final_indexes, first_iteration_index1, remaining_toks1, ques_terms, answer_terms, justifications)  ## second iteration

    ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
    return Final_indexes, ROCC_ranked_PRF



def get_iterative_alignment_justifications_unsupervised_semantic_drift(ques_text, answer_text, justifications, IDF_vals, embedding_index, gold_justification_indexes, emb_size=100, subgraph_size=5):  ## Alignment over embeddings for sentence selection
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)
    Final_indexes = []

    ###### First iteration is here
    first_iteration_index1, remaining_toks1, remaining_toks2 = get_alignment_justification(ques_terms + answer_terms, [], justifications, embedding_index, emb_size, IDF_vals)

    Final_indexes += [first_iteration_index1[0]]  # , first_iteration_index1[1]]
    selected_just_toks = Preprocess_QA_sentences(justifications[first_iteration_index1[0]], 1)

    new_query_terms = remaining_toks1 + selected_just_toks    ## if we simply concatenate the retrieved justification, it will lead to semantic drift.

    # new_justification1 = justifications[:first_iteration_index1] + justifications[first_iteration_index1+1:]
    second_iteration_index1, remaining_toks1_2, remaining_toks2_2 = get_alignment_justification(new_query_terms, [], justifications, embedding_index, emb_size, IDF_vals)

    for i1 in second_iteration_index1:
        if i1 in Final_indexes:
            pass
        else:
            Final_indexes.append(i1)
            break
    # else:
    # Final_indexes += [first_iteration_index1[1]]
    selected_just_toks = Preprocess_QA_sentences(justifications[second_iteration_index1[0]], 1)

    new_query_terms = remaining_toks1_2 + selected_just_toks
        # print ("yes, we do have some really short queries. ")

    # new_justification1 = justifications[:first_iteration_index2] + justifications[first_iteration_index2 + 1:]
    second_iteration_index1, remaining_toks1_3, remaining_toks2_3 = get_alignment_justification(new_query_terms, [], justifications, embedding_index, emb_size, IDF_vals)

    for i1 in second_iteration_index1:
        if i1 in Final_indexes:
            pass
        else:
            Final_indexes.append(i1)
            break

    selected_just_toks = Preprocess_QA_sentences(justifications[second_iteration_index1[0]], 1)

    new_query_terms = remaining_toks1_3 + selected_just_toks
    # print ("yes, we do have some really short queries. ")

    # new_justification1 = justifications[:first_iteration_index2] + justifications[first_iteration_index2 + 1:]
    second_iteration_index1, remaining_toks1_4, remaining_toks2_4 = get_alignment_justification(new_query_terms, [],
                                                                                                justifications,
                                                                                                embedding_index,
                                                                                                emb_size, IDF_vals)

    for i1 in second_iteration_index1:
        if i1 in Final_indexes:
            pass
        else:
            Final_indexes.append(i1)
            break

    ROCC_ranked_PRF = F1_Score_individual_just(gold_justification_indexes, Final_indexes)
    # print ("the final indices look like ", Final_indexes)
    return Final_indexes, ROCC_ranked_PRF  ## returning the whole passage



# def get_BM25_subgraph(ques_text, answer_text, justifications, subgraph_size):  ## subgraph size is not used
#
#     gensim_BM25_scores_indexes = np.argsort(get_BM25_scores(justifications, ques_text + " " + answer_text))[::-1]
#     top_ranked_passage = ""
#     for i in gensim_BM25_scores_indexes[:subgraph_size]:
#         top_ranked_passage += justifications[i] + " "
#
#     return top_ranked_passage, gensim_BM25_scores_indexes[:subgraph_size]  ## returning the whole passage


def get_BM25_subgraph(ques_text, answer_text, justifications):  ## subgraph size is not used

    gensim_BM25_scores_indexes = np.argsort(get_BM25_scores(justifications, ques_text + " " + answer_text))[::-1]

    return gensim_BM25_scores_indexes  ## returning the whole passage






