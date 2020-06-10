
import numpy as np
import collections
from Overlap_analysis import calculate_overlap, calculate_all_overlap, calculate_overlap_labels, get_union, get_intersection, get_intersection_withIDF, calculate_kappa, calculate_alignment_overlap \
                            , calculate_alignment_union
# from Compute_F1 import mean_confidence_interval, meta_voting_ensemble, meta_voting_ensemble_BECKY
from itertools import combinations
from Compute_F1 import get_differences_list
import math
######################################## These functions are to implement 2^n combinations of subgraphs and select the best one out of it.
def get_all_combination_best_graph(pred_labels_over_runs, performance_runs):  ## Edge based model
    runs = list(pred_labels_over_runs.keys())
    meta_subgraphs = []
    for i in range(len(runs)-1):
        meta_subgraphs += list(combinations(runs, i+2))
    print ("len of meta subgraphs are ", len(meta_subgraphs))
    meta_graph_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_score = []
        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):
            for rk2 in meta_sub_graph1[ik1+1:]:
                current_subgraph_score.append ( performance_runs[rk1]+performance_runs[rk2] / float(calculate_overlap_labels(pred_labels_over_runs[rk1], pred_labels_over_runs[rk2]) ) )
        meta_graph_scores.append(sum(current_subgraph_score)/float(len(current_subgraph_score)))  ## taking average of subgraph scores

    print ("the len of meta graph scores are : ", len(meta_graph_scores))
    best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
    print ("best subgraph is: ", meta_subgraphs[best_sub_graph_index], max(meta_graph_scores))

    return meta_subgraphs[best_sub_graph_index]

def get_all_combination_Vikas_EdgeAPPROACH_withCoverage_best_graph(pred_labels_over_runs, performance_runs, gold_labels):
    runs = list(pred_labels_over_runs.keys())
    meta_subgraphs = []
    for i in range(len(runs)-1):
        meta_subgraphs += list(combinations(runs, i+2))
    print ("len of meta subgraphs are ", len(meta_subgraphs))
    meta_graph_scores = []
    meta_graph_coverage_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_score = []
        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
               prediction_coverage = pred_labels_over_runs[rk1]

            for rk2 in meta_sub_graph1[ik1+1:]:
                current_subgraph_score.append(performance_runs[rk1] + performance_runs[rk2] / float(calculate_overlap_labels(pred_labels_over_runs[rk1], pred_labels_over_runs[rk2])))
                prediction_coverage = get_union(prediction_coverage, pred_labels_over_runs[rk2])

        final_coverage = sum(get_intersection(prediction_coverage, gold_labels))/float(sum(gold_labels))
        meta_graph_coverage_scores.append(final_coverage)
        meta_graph_scores.append( (sum(current_subgraph_score)/float(len(current_subgraph_score)) ) * final_coverage )  ## taking average of subgraph scores

    print ("the len of meta graph scores are : ", len(meta_graph_scores))
    best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
    print ("best subgraph is: ", meta_subgraphs[best_sub_graph_index], max(meta_graph_scores),meta_graph_coverage_scores[best_sub_graph_index])

    return meta_subgraphs[best_sub_graph_index]




def get_all_combination_STEVE_best_graph(pred_labels_over_runs, performance_runs): ## steve's suggestions
    runs = list(pred_labels_over_runs.keys())
    meta_subgraphs = []
    for i in range(len(runs)-1):
        meta_subgraphs += list(combinations(runs, i+2))
    print ("len of meta subgraphs are ", len(meta_subgraphs))


    meta_graph_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):
            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1+1:]:
                current_subgraph_overlap.append (float(calculate_overlap_labels(pred_labels_over_runs[rk1], pred_labels_over_runs[rk2]) ) )
        avg_score =  sum(current_subgraph_perf)/float(len(current_subgraph_perf))
        avg_overlap =  sum(current_subgraph_overlap)/float(len(current_subgraph_overlap))

        meta_graph_scores.append(avg_score/float(avg_overlap))  ## taking average of subgraph scores

    print ("the len of meta graph scores are : ", len(meta_graph_scores))
    best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
    print ("best subgraph is: ", meta_subgraphs[best_sub_graph_index], max(meta_graph_scores))

    return meta_subgraphs[best_sub_graph_index]





def get_all_combination_withCoverage_best_graph(KB_terms, performance_runs, Ques_terms, Ans_terms):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    # for i in range(len(runs)-1):
    #     meta_subgraphs += list(combinations(runs, i+2))

    meta_subgraphs += list(combinations(runs, 4))

    meta_graph_scores = []
    meta_graph_coverage_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
               prediction_coverage = KB_terms[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1+1:-1]:  ##### This is equivalent to M C 2

               current_subgraph_overlap.append (float(calculate_overlap(KB_terms[rk1], KB_terms[rk2]) ) )
               prediction_coverage = get_union(prediction_coverage, KB_terms[rk2])

        avg_score =  sum(current_subgraph_perf)/float(len(current_subgraph_perf))
        avg_overlap =  sum(current_subgraph_overlap)/float(len(current_subgraph_overlap))
        # print ("the ")
        final_coverage = len(get_intersection(prediction_coverage, gold_labels))/float(len(gold_labels))
        meta_graph_coverage_scores.append(final_coverage)
        # meta_graph_scores.append( (avg_score/float(avg_overlap+1)) * final_coverage )  ## taking average of subgraph scores
        meta_graph_scores.append( avg_score * final_coverage )  ## taking average of subgraph scores

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
        return meta_subgraphs[best_sub_graph_index]
    except ValueError:
        return "Crashed"

#######################################



def get_all_combination_withCoverage_best_graph_Cand_boost(KB_terms, performance_runs, Ques_terms, Ans_terms, subgraph_size):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    for i in range(subgraph_size-2):
        meta_subgraphs += list(combinations(runs, i+2))

    # for i in range(subgraph_size):  ## for taking best subgraph amongst subgraphs of size 3,4,5
    #     meta_subgraphs += list(combinations(runs, i+3))

    # meta_subgraphs += list(combinations(runs, subgraph_size))

    meta_graph_scores = []
    meta_graph_coverage_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
                prediction_coverage = KB_terms[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1 + 1:-1]:  ##### This is equivalent to M C 2

                current_subgraph_overlap.append(float(calculate_overlap(KB_terms[rk1], KB_terms[rk2])))
                prediction_coverage = get_union(prediction_coverage, KB_terms[rk2])

        avg_score = sum(current_subgraph_perf) / float(len(current_subgraph_perf))
        avg_overlap = sum(current_subgraph_overlap) / float(max(1,len(current_subgraph_overlap)))
        # print ("the ")
        final_query_coverage = len(get_intersection(prediction_coverage, Ques_terms)) / max(1,float(len(Ques_terms)))
        final_ans_coverage = len(get_intersection(prediction_coverage, Ans_terms)) / max(1,float(len(Ans_terms)))

        meta_graph_coverage_scores.append(final_query_coverage)
        # meta_graph_scores.append( avg_score  * final_ans_coverage * final_query_coverage)  ## taking average of subgraph scores
        # if subgraph_size>2:
        #    print ("the avg score, overlap and coverage looks like: ", avg_score, avg_overlap, final_query_coverage, final_ans_coverage)
        # meta_graph_scores.append( (avg_score/float(1+avg_overlap))  * (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores
        meta_graph_scores.append( avg_score * (1+12*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))

        return meta_subgraphs[best_sub_graph_index]
    except ValueError:
        return "Crashed"

#######################################


#######################################



def get_all_combination_withCoverage_best_graph_Cand_boost_withIDF(KB_terms, performance_runs, Ques_terms, Ans_terms, subgraph_size, IDF_vals):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    for i in range(subgraph_size-1):
        meta_subgraphs += list(combinations(runs, i+2))

    # for i in range(subgraph_size):  ## for taking best subgraph amongst subgraphs of size 3,4,5
    #     meta_subgraphs += list(combinations(runs, i+3))

    # meta_subgraphs += list(combinations(runs, subgraph_size))

    meta_graph_scores = []
    meta_graph_coverage_scores = []
    meta_graph_ans_coverage_scores = []
    meta_graph_overlap_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
                prediction_coverage = KB_terms[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1 + 1:-1]:  ##### This is equivalent to M C 2

                current_subgraph_overlap.append(float(calculate_overlap(KB_terms[rk1], KB_terms[rk2])))
                prediction_coverage = get_union(prediction_coverage, KB_terms[rk2])

        avg_score = sum(current_subgraph_perf) / float(len(current_subgraph_perf))
        avg_overlap = sum(current_subgraph_overlap) / float(max(1,len(current_subgraph_overlap)))
        # print ("the ")
        final_query_coverage = get_intersection_withIDF(prediction_coverage, Ques_terms, IDF_vals) / max(1,float(len(Ques_terms)))
        final_ans_coverage = get_intersection_withIDF(prediction_coverage, Ans_terms, IDF_vals) / max(1,float(len(Ans_terms)))

        meta_graph_coverage_scores.append(final_query_coverage)
        meta_graph_ans_coverage_scores.append(final_ans_coverage)
        meta_graph_overlap_scores.append(avg_overlap)
        # meta_graph_scores.append( avg_score  * final_ans_coverage * final_query_coverage)  ## taking average of subgraph scores
        # if subgraph_size>2:
        #    print ("the avg score, overlap and coverage looks like: ", avg_score, avg_overlap, final_query_coverage, final_ans_coverage)
        # meta_graph_scores.append( (avg_score/float(1+avg_overlap))  * (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores
        meta_graph_scores.append( (1+avg_score/float(1+avg_overlap)) *  (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores ##  *  * # 1+avg_overlap

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
        # print ("checking weather this returns any overlap val or not ", meta_graph_overlap_scores)
        return meta_subgraphs[best_sub_graph_index], meta_graph_overlap_scores[best_sub_graph_index], meta_graph_coverage_scores[best_sub_graph_index], meta_graph_ans_coverage_scores[best_sub_graph_index]
    except ValueError:
        return "Crashed"

#######################################





#####################################


def get_all_combination_withCoverage_Alignment_IDF(Justification_ans_scores, ans_IDF_mat, Justification_ques_scores, ques_IDF_mat, Justification_ques_ans_scores_together, performance_runs, Ques_terms, Ans_terms, subgraph_size, IDF_vals):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    for i in range(subgraph_size):
        meta_subgraphs += list(combinations(runs, i+2))

    # for i in range(subgraph_size):  ## for taking best subgraph amongst subgraphs of size 3,4,5
    #     meta_subgraphs += list(combinations(runs, i+3))

    # meta_subgraphs += list(combinations(runs, subgraph_size))

    meta_graph_scores = []
    meta_graph_coverage_scores = []
    meta_graph_ans_coverage_scores = []
    meta_graph_overlap_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
                ques_coverage = Justification_ques_scores[rk1]
                ans_coverage = Justification_ans_scores[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1 + 1:-1]:  ##### This is equivalent to M C 2
                if len(Justification_ques_ans_scores_together[rk1]) > 0 and len(Justification_ques_ans_scores_together[rk2]) > 0 :
                    # if len(Justification_ques_ans_scores_together[rk1])>0 and len(Justification_ques_ans_scores_together[rk2]) > 0:
                        if len(Justification_ques_ans_scores_together[rk1]) == len(Justification_ques_ans_scores_together[rk2]):
                            current_subgraph_overlap.append(float(calculate_alignment_overlap(Justification_ques_ans_scores_together[rk1], Justification_ques_ans_scores_together[rk2])))
                            ques_coverage = calculate_alignment_union(ques_coverage, Justification_ques_scores[rk2])
                            ans_coverage = calculate_alignment_union(ans_coverage, Justification_ans_scores[rk2])
                        else:
                            print ("find out why this is happening", len(Justification_ques_ans_scores_together[rk1]), len(Justification_ques_ans_scores_together[rk2]))

                else:
                    pass

        avg_score = sum(current_subgraph_perf) / float(len(current_subgraph_perf))
        avg_overlap = sum(current_subgraph_overlap) / float(max(1,len(current_subgraph_overlap)))

        # final_query_coverage = get_intersection_withIDF(prediction_coverage, Ques_terms, IDF_vals) / max(1,float(len(Ques_terms)))
        # final_ans_coverage = get_intersection_withIDF(prediction_coverage, Ans_terms, IDF_vals) / max(1,float(len(Ans_terms)))
        # if len(ques_coverage) > 0 and len(ans_coverage) > 0 :
        if len(ques_coverage) == len(ques_IDF_mat):
           ques_coverage = [a*b for a,b in zip(ques_IDF_mat, ques_coverage)]
           # print ("yes, we do come here ")
        # else:
        #    print ("The len of cov vector and idf vector are different, checkout why", len(ques_coverage), len(ques_IDF_mat))
        if len(ans_coverage) == len(ans_IDF_mat):
           ans_coverage = [a*b for a,b in zip(ans_IDF_mat, ans_coverage)]
        # else:
        #     print ("yep, we had these cases: ")
        final_query_coverage = (sum(ques_coverage)) / float(max(1, len(ques_coverage)))
        final_ans_coverage = (sum(ans_coverage)) / float(max(1, len(ans_coverage)))

        # final_query_coverage = math.log(max(1,sum(ques_coverage)))/float(max(1,len(ques_coverage)))
        # final_ans_coverage = math.log(max(1,sum(ans_coverage)))/float(max(1,len(ans_coverage)))

        meta_graph_coverage_scores.append(final_query_coverage)
        meta_graph_ans_coverage_scores.append(final_ans_coverage)
        meta_graph_overlap_scores.append(avg_overlap)
        # meta_graph_scores.append( avg_score  * final_ans_coverage * final_query_coverage)  ## taking average of subgraph scores
        # if subgraph_size>2:
        #    print ("the avg score, overlap and coverage looks like: ", avg_score, avg_overlap, final_query_coverage, final_ans_coverage)
        # meta_graph_scores.append( (1+avg_score)  * (1+1*final_ans_coverage) * (1+final_query_coverage) )   ## taking average of subgraph scores
        meta_graph_scores.append( ((1+avg_score)/float(1+0)) *  (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores ##  *  * # 1+avg_overlap

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
        list_of_top_subgraphs = list(np.argsort(meta_graph_scores))[::-1]
        vicinity_scores = []
        for top_graph_ind in list_of_top_subgraphs[0:20]:
            vicinity_scores.append(sum(get_differences_list(meta_subgraphs[top_graph_ind])))
        # print ("checking weather this returns any overlap val or not ", meta_graph_overlap_scores)
        # print("The first calculated index was: ", best_sub_graph_index, "then", list_of_top_subgraphs[vicinity_scores.index(min(vicinity_scores))])
        return meta_subgraphs[list_of_top_subgraphs[vicinity_scores.index(min(vicinity_scores))]], meta_graph_overlap_scores[best_sub_graph_index], meta_graph_coverage_scores[best_sub_graph_index], meta_graph_ans_coverage_scores[best_sub_graph_index]

        # return meta_subgraphs[best_sub_graph_index], meta_graph_overlap_scores[best_sub_graph_index], meta_graph_coverage_scores[best_sub_graph_index], meta_graph_ans_coverage_scores[best_sub_graph_index]
        # return meta_subgraphs[best_sub_graph_index], meta_graph_overlap_scores[best_sub_graph_index], meta_graph_coverage_scores[best_sub_graph_index], meta_graph_ans_coverage_scores[best_sub_graph_index]
    except ValueError:
        return "Crashed"

#######################################



#####################################


def get_all_combination_withCoverage_Alignment_Regression(Justification_ans_scores, ans_IDF_mat, Justification_ques_scores, ques_IDF_mat, Justification_ques_ans_scores_together, performance_runs, Ques_terms, Ans_terms, subgraph_size, IDF_vals, n_top_ranked_sets):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    for i in range(subgraph_size):
        meta_subgraphs += list(combinations(runs, i+2))

    # for i in range(subgraph_size):  ## for taking best subgraph amongst subgraphs of size 3,4,5
    #     meta_subgraphs += list(combinations(runs, i+3))

    # meta_subgraphs += list(combinations(runs, subgraph_size))

    meta_graph_scores = []
    meta_graph_coverage_scores = []
    meta_graph_ans_coverage_scores = []
    meta_graph_overlap_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
                ques_coverage = Justification_ques_scores[rk1]
                ans_coverage = Justification_ans_scores[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1 + 1:-1]:  ##### This is equivalent to M C 2
                if len(Justification_ques_ans_scores_together[rk1]) > 0 and len(Justification_ques_ans_scores_together[rk2]) > 0 :
                    # if len(Justification_ques_ans_scores_together[rk1])>0 and len(Justification_ques_ans_scores_together[rk2]) > 0:
                        if len(Justification_ques_ans_scores_together[rk1]) == len(Justification_ques_ans_scores_together[rk2]):
                            current_subgraph_overlap.append(float(calculate_alignment_overlap(Justification_ques_ans_scores_together[rk1], Justification_ques_ans_scores_together[rk2])))
                            ques_coverage = calculate_alignment_union(ques_coverage, Justification_ques_scores[rk2])
                            ans_coverage = calculate_alignment_union(ans_coverage, Justification_ans_scores[rk2])
                        else:
                            print ("find out why this is happening", len(Justification_ques_ans_scores_together[rk1]), len(Justification_ques_ans_scores_together[rk2]))

                else:
                    pass

        avg_score = sum(current_subgraph_perf) / float(len(current_subgraph_perf))
        avg_overlap = sum(current_subgraph_overlap) / float(max(1,len(current_subgraph_overlap)))

        # final_query_coverage = get_intersection_withIDF(prediction_coverage, Ques_terms, IDF_vals) / max(1,float(len(Ques_terms)))
        # final_ans_coverage = get_intersection_withIDF(prediction_coverage, Ans_terms, IDF_vals) / max(1,float(len(Ans_terms)))
        # if len(ques_coverage) > 0 and len(ans_coverage) > 0 :
        if len(ques_coverage) == len(ques_IDF_mat):
           ques_coverage = [a*b for a,b in zip(ques_IDF_mat, ques_coverage)]
           # print ("yes, we do come here ")
        # else:
        #    print ("The len of cov vector and idf vector are different, checkout why", len(ques_coverage), len(ques_IDF_mat))
        if len(ans_coverage) == len(ans_IDF_mat):
           ans_coverage = [a*b for a,b in zip(ans_IDF_mat, ans_coverage)]
        # else:
        #     print ("yep, we had these cases: ")
        final_query_coverage = (sum(ques_coverage)) / float(max(1, len(ques_coverage)))
        final_ans_coverage = (sum(ans_coverage)) / float(max(1, len(ans_coverage)))

        # final_query_coverage = math.log(max(1,sum(ques_coverage)))/float(max(1,len(ques_coverage)))
        # final_ans_coverage = math.log(max(1,sum(ans_coverage)))/float(max(1,len(ans_coverage)))

        meta_graph_coverage_scores.append(final_query_coverage)
        meta_graph_ans_coverage_scores.append(final_ans_coverage)
        meta_graph_overlap_scores.append(avg_overlap)


        # vicinity_score = sum(get_differences_list(meta_sub_graph1))/float(len(get_differences_list(meta_sub_graph1)))
        # vicinity_score = min(get_differences_list(meta_sub_graph1))
        meta_graph_scores.append( ((1+avg_score) / float(1+0)) *  (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores ##  *  * # 1+avg_overlap

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
        list_of_top_subgraphs = list(np.argsort(meta_graph_scores))[::-1]

        """
        ## adding vicinity_scores_here:
        top_ranked_by_ROCC_meta_subgraphs = [meta_subgraphs[i1] for i1 in list_of_top_subgraphs[0:100]]
        vicinity_scores = []
        for top_graph_1 in top_ranked_by_ROCC_meta_subgraphs:
            vicinity_scores.append(sum(get_differences_list(top_graph_1))/float(len(get_differences_list(top_graph_1))))

        list_of_top_subgraphs_from_vicinity_scores = list(np.argsort(vicinity_scores))[::-1]

        top_10percent_subgraphs = []
        # for top_subgraph1 in list_of_top_subgraphs[0:math.floor(0.1*len(list_of_top_subgraphs))]:
        for top_subgraph1 in list_of_top_subgraphs_from_vicinity_scores[0:n_top_ranked_sets]:
            top_10percent_subgraphs.append(top_ranked_by_ROCC_meta_subgraphs[top_subgraph1])

        """

        top_10percent_subgraphs = []
        # for top_subgraph1 in list_of_top_subgraphs[0:math.floor(0.1*len(list_of_top_subgraphs))]:
        for top_subgraph1 in list_of_top_subgraphs[0:n_top_ranked_sets]:
            top_10percent_subgraphs.append(meta_subgraphs[top_subgraph1])
        # print ("checking weather this returns any overlap val or not ", meta_graph_overlap_scores)
        # print("The first calculated index was: ", best_sub_graph_index, "then", list_of_top_subgraphs[vicinity_scores.index(min(vicinity_scores))])


        return meta_subgraphs[best_sub_graph_index], top_10percent_subgraphs, meta_graph_overlap_scores[best_sub_graph_index], meta_graph_coverage_scores[best_sub_graph_index], meta_graph_ans_coverage_scores[best_sub_graph_index]

    except ValueError:
        return "Crashed"



#####################################


def get_all_combination_withCoverage_SOFT_Alignment_IDF(Justification_ans_scores, ans_IDF_mat, Justification_ques_scores, ques_IDF_mat, Justification_ques_ans_scores_together, performance_runs, Ques_terms, Ans_terms, subgraph_size, IDF_vals):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    for i in range(subgraph_size):
        meta_subgraphs += list(combinations(runs, i+2))

    # for i in range(subgraph_size):  ## for taking best subgraph amongst subgraphs of size 3,4,5
    #     meta_subgraphs += list(combinations(runs, i+3))

    # meta_subgraphs += list(combinations(runs, subgraph_size))

    meta_graph_scores = []
    meta_graph_coverage_scores = []
    meta_graph_ans_coverage_scores = []
    meta_graph_overlap_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
                ques_coverage = Justification_ques_scores[rk1]
                ans_coverage = Justification_ans_scores[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1 + 1:-1]:  ##### This is equivalent to M C 2
                if len(Justification_ques_ans_scores_together[rk1]) > 0 and len(Justification_ques_ans_scores_together[rk2]) > 0 :
                    if len(Justification_ques_ans_scores_together[rk1]) == len(Justification_ques_ans_scores_together[rk2]):
                        current_subgraph_overlap.append(float(calculate_alignment_overlap(Justification_ques_ans_scores_together[rk1], Justification_ques_ans_scores_together[rk2])))
                        ques_coverage = calculate_alignment_union(ques_coverage, Justification_ques_scores[rk2])
                        ans_coverage = calculate_alignment_union(ans_coverage, Justification_ans_scores[rk2])
                    else:
                        print ("find out why this is happening", len(Justification_ques_ans_scores_together[rk1]), len(Justification_ques_ans_scores_together[rk2]))

                else:
                    pass

        avg_score = sum(current_subgraph_perf) / float(len(current_subgraph_perf))
        avg_overlap = sum(current_subgraph_overlap) / float(max(1,len(current_subgraph_overlap)))

        # final_query_coverage = get_intersection_withIDF(prediction_coverage, Ques_terms, IDF_vals) / max(1,float(len(Ques_terms)))
        # final_ans_coverage = get_intersection_withIDF(prediction_coverage, Ans_terms, IDF_vals) / max(1,float(len(Ans_terms)))
        # if len(ques_coverage) > 0 and len(ans_coverage) > 0 :
        if len(ques_coverage) == len(ques_IDF_mat):
           ques_coverage = [a*b for a,b in zip(ques_IDF_mat, ques_coverage)]
           # print ("yes, we do come here ")
        # else:
        #    print ("The len of cov vector and idf vector are different, checkout why", len(ques_coverage), len(ques_IDF_mat))
        if len(ans_coverage) == len(ans_IDF_mat):
           ans_coverage = [a*b for a,b in zip(ans_IDF_mat, ans_coverage)]
        # else:
        #     print ("yep, we had these cases: ")
        final_query_coverage = (sum(ques_coverage)) / float(max(1, len(ques_coverage)*len(current_subgraph_overlap)))
        final_ans_coverage = (sum(ans_coverage)) / float(max(1, len(ans_coverage)*len(current_subgraph_overlap)))

        # final_query_coverage = math.log(max(1,sum(ques_coverage)))/float(max(1,len(ques_coverage)))
        # final_ans_coverage = math.log(max(1,sum(ans_coverage)))/float(max(1,len(ans_coverage)))

        meta_graph_coverage_scores.append(final_query_coverage)
        meta_graph_ans_coverage_scores.append(final_ans_coverage)
        meta_graph_overlap_scores.append(avg_overlap)
        # meta_graph_scores.append( avg_score  * final_ans_coverage * final_query_coverage)  ## taking average of subgraph scores
        # if subgraph_size>2:
        #    print ("the avg score, overlap and coverage looks like: ", avg_score, avg_overlap, final_query_coverage, final_ans_coverage)
        # meta_graph_scores.append( (1+avg_score)  * (1+1*final_ans_coverage) * (1+final_query_coverage) )   ## taking average of subgraph scores
        meta_graph_scores.append( ((1+avg_score)/float(1)) *  (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores ##  *  * # 1+avg_overlap

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
        # print ("checking weather this returns any overlap val or not ", meta_graph_overlap_scores)
        return meta_subgraphs[best_sub_graph_index], meta_graph_overlap_scores[best_sub_graph_index], meta_graph_coverage_scores[best_sub_graph_index], meta_graph_ans_coverage_scores[best_sub_graph_index]
    except ValueError:
        return "Crashed"

#######################################






#######################################



def get_all_combination_withCoverage_best_graph_Cand_boost_ALL(KB_terms, performance_runs, Ques_terms, Ans_terms):  ## gold_labels_list is QA terms and pred_labels_over_runs is justification terms
    runs = list(performance_runs.keys())
    gold_labels = Ques_terms + Ans_terms
    # print("the gold_labels list looks like: ", runs)
    meta_subgraphs = []

    for i in range(len(runs)-1):
        meta_subgraphs += list(combinations(runs, i+2))

    # meta_subgraphs += list(combinations(runs, subgraph_size))

    meta_graph_scores = []
    meta_graph_coverage_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_overlap = []
        current_subgraph_perf = []

        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

            if ik1 == 0:  ## initializing the coverage list
                prediction_coverage = KB_terms[rk1]

            current_subgraph_perf.append(performance_runs[rk1])
            for rk2 in meta_sub_graph1[ik1 + 1:-1]:  ##### This is equivalent to M C 2

                current_subgraph_overlap.append(float(calculate_overlap(KB_terms[rk1], KB_terms[rk2])))
                prediction_coverage = get_union(prediction_coverage, KB_terms[rk2])

        avg_score = sum(current_subgraph_perf) / float(len(current_subgraph_perf))
        avg_overlap = sum(current_subgraph_overlap) / float(max(1,len(current_subgraph_overlap)))
        # print ("the ")
        final_query_coverage = len(get_intersection(prediction_coverage, Ques_terms)) / float(len(Ques_terms))
        final_ans_coverage = len(get_intersection(prediction_coverage, Ans_terms)) / float(len(Ans_terms))

        meta_graph_coverage_scores.append(final_query_coverage)
        # meta_graph_scores.append( avg_score  * final_ans_coverage * final_query_coverage)  ## taking average of subgraph scores
        # if subgraph_size>2:
        #    print ("the avg score, overlap and coverage looks like: ", avg_score, avg_overlap, final_query_coverage, final_ans_coverage)
        meta_graph_scores.append( (avg_score/float(1+avg_overlap))  * (1+1*final_ans_coverage) * (1+final_query_coverage) )  ## taking average of subgraph scores

    # print ("the len of meta graph scores are : ", len(meta_graph_scores))
    try:
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))

        return meta_subgraphs[best_sub_graph_index]
    except ValueError:
        return "Crashed"

#######################################



def get_all_combination_forN_sizes_withCoverage_best_graph(pred_labels_over_runs, all_prediction_label_runs, performance_runs, gold_labels_list, BiNODE_overlap, mean_score):
    runs = list(pred_labels_over_runs.keys())
    best_subgraphs_diff_sizes = {}  ### (P/O)*C
    best_subgraphs_overlaps = {}  ## just 1/O factor
    best_subgraphs_Perf_Over = {}  ## Just (P/O) factor, no coverage

    gold_labels = list(range(len(gold_labels_list)))

    all_subgraphs = []
    feature_x = []
    label_y = []

    POC_score = 0
    POC_subgraph = []

    for i in range(len(runs)-1):
        meta_subgraphs = list(combinations(runs, i+2))

        meta_graph_scores = []  ### same sequence as above
        meta_graph_Overlap_scores = []
        meta_graph_PERF_Overlap_scores = []


        meta_graph_coverage_scores = []

        for meta_sub_graph1 in meta_subgraphs:
            current_subgraph_overlap = []
            current_subgraph_perf = []

            for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):

                if ik1 == 0:  ## initializing the coverage list
                   prediction_coverage = pred_labels_over_runs[rk1]

                current_subgraph_perf.append(performance_runs[rk1])
                for rk2 in meta_sub_graph1[ik1+1:]:
                    current_subgraph_overlap.append (BiNODE_overlap[str(rk1)+str(rk2)] )
                    prediction_coverage = get_union(prediction_coverage, pred_labels_over_runs[rk2])

            avg_score =  sum(current_subgraph_perf)/float(len(current_subgraph_perf))
            avg_overlap =  sum(current_subgraph_overlap)/float(len(current_subgraph_overlap))
            # print ("the ")
            final_coverage = len(get_intersection(prediction_coverage, gold_labels))/float(len(gold_labels))
            meta_graph_coverage_scores.append(final_coverage)
            meta_graph_scores.append( (avg_score/float(avg_overlap)) * final_coverage )  ## taking average of subgraph scores

            ############### for linear regression statistics and feature generation
            # feature_x.append([avg_score, 1/float(avg_overlap), final_coverage, avg_score/float(avg_overlap),(avg_score/float(avg_overlap))*final_coverage, avg_score*final_coverage])
            feature_x.append([avg_score, avg_overlap, final_coverage])
            best_subgraph_preds = {mn1: all_prediction_label_runs[mn1] for mn1 in meta_sub_graph1}
            subgraph_ensemble_performance = meta_voting_ensemble(best_subgraph_preds, gold_labels_list, math.ceil(len(meta_sub_graph1) / 2))
            # print("the subgraph ensemble performance looks like:  ", subgraph_ensemble_performance)

            label_y.append(subgraph_ensemble_performance - mean_score)
            all_subgraphs.append(meta_sub_graph1)

            ###################

            meta_graph_Overlap_scores.append(1/float(avg_overlap))
            meta_graph_PERF_Overlap_scores.append(avg_score/float(avg_overlap))
        # print ("the len of meta graph scores are : ", len(meta_graph_scores))
        best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))

        if max(meta_graph_scores)>POC_score:
           POC_score = max(meta_graph_scores)
           POC_subgraph =  meta_subgraphs[best_sub_graph_index]

        print ("best subgraph is: ", meta_subgraphs[best_sub_graph_index], max(meta_graph_scores),meta_graph_coverage_scores[best_sub_graph_index])
        best_subgraphs_diff_sizes.update({i+2: meta_subgraphs[best_sub_graph_index]})
        best_subgraphs_overlaps.update({i+2: meta_subgraphs[meta_graph_Overlap_scores.index(max(meta_graph_Overlap_scores))]})
        best_subgraphs_Perf_Over.update({i+2:meta_subgraphs[meta_graph_PERF_Overlap_scores.index(max(meta_graph_PERF_Overlap_scores))]})


    return best_subgraphs_diff_sizes, best_subgraphs_overlaps, best_subgraphs_Perf_Over, feature_x, label_y, all_subgraphs, POC_subgraph



########################################

def get_ensemble_perf_based_best_subgraph(meta_nodes, prediction_label_runs, gold_labels, best_performance, final_best_graph):

    best_subgraph_preds = {mn1: prediction_label_runs[mn1] for mn1 in meta_nodes}
    meta_ensemble_performance = meta_voting_ensemble(best_subgraph_preds, gold_labels, math.ceil(len(meta_nodes) / 2))

    # print("the final meta ensemble performance is: ", meta_ensemble_performance)

    if meta_ensemble_performance > best_performance:
        final_best_graph = meta_nodes
        best_performance = meta_ensemble_performance
    return best_performance, final_best_graph



########################################


def get_complete_graph(pred_labels_over_runs, performance_runs, subgraph_size=4):
    runs = list(pred_labels_over_runs.keys())

    meta_subgraphs = list(combinations(runs, subgraph_size))
    print ("len of meta subgraphs are ", len(meta_subgraphs))


    meta_graph_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_score = []
        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):
            for rk2 in meta_sub_graph1[ik1+1:]:
                current_subgraph_score.append ( performance_runs[rk1]*performance_runs[rk2] / float(calculate_overlap_labels(pred_labels_over_runs[rk1], pred_labels_over_runs[rk2]) ) )
        meta_graph_scores.append(sum(current_subgraph_score)/float(len(current_subgraph_score)))

    print ("the len of meta graph scores are : ", len(meta_graph_scores))
    best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
    print ("best subgraph is: ", meta_subgraphs[best_sub_graph_index], max(meta_graph_scores))

    return meta_subgraphs[best_sub_graph_index]



def get_unsupervised_sub_graph(pred_labels_over_runs, subgraph_size=4):
    runs = list(pred_labels_over_runs.keys())

    meta_subgraphs = list(combinations(runs, subgraph_size))
    print ("len of meta subgraphs are ", len(meta_subgraphs))


    meta_graph_scores = []

    for meta_sub_graph1 in meta_subgraphs:
        current_subgraph_score = []
        for ik1, rk1 in enumerate(meta_sub_graph1[:-1]):
            for rk2 in meta_sub_graph1[ik1+1:]:
                current_subgraph_score.append ( 1 / float(calculate_overlap_labels(pred_labels_over_runs[rk1], pred_labels_over_runs[rk2]) ) )
        meta_graph_scores.append(sum(current_subgraph_score))

    print ("the len of meta graph scores are : ", len(meta_graph_scores))
    best_sub_graph_index = meta_graph_scores.index(max(meta_graph_scores))
    print ("best subgraph is: ", meta_subgraphs[best_sub_graph_index], max(meta_graph_scores))

    return meta_subgraphs[best_sub_graph_index]




## dummy function - complete this later
def get_node_pair_score(runs):

    for ik1, rk1 in enumerate(runs[:-1]):
        for rk2 in runs[ik1+1:]:
            meta_graph.update({rk1+" " + rk2 : calculate_overlap_labels(pred_labels_over_runs[rk1], pred_labels_over_runs[rk2]) })

