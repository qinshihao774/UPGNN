## In[BaseEvaluation]
from abc import ABC, abstractmethod
class BaseEvaluation(ABC):
    """Base class for evaluations that provide a score metric"""
    @abstractmethod
    def get_score(self, explanations):
        """
        Returns the score of the metric
        :param explanations: list of explanations by the explainer
        :return: score
        """
        pass

## In[AUCEvaluation]
class AUCEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.
    
    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.
    
    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth

    def get_score(self, explanations):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :return: auc score
        """
        return evaluation_auc_node(explanations, self.ground_truth)

## In[Classification accuray]
def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc

## In[Explanation auc]
import numpy as np
from sklearn.metrics import roc_auc_score
def evaluation_auc_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.
    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node

        ground_truth_node = []
        prediction_node = []

        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i].numpy()
            idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score

## In[roc-auc]
def roc_auc(exp, gt):
    score = roc_auc_score(gt,exp)
    return score