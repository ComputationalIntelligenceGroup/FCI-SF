import numpy as np

from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph


class EndpointConfusion:
    """
    Compute the arrow confusion between two graphs.
    """
    __Fp = 0
    __Fn = 0
    __Tp = 0
    __Tn = 0

    __FpCE = 0
    __FnCE = 0
    __TpCE = 0
    __TnCE = 0

    def __init__(self, truth: Graph, est: Graph, endpoint: Endpoint):
        """
        Compute and store the arrow confusion between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est :
            Estimated graph.
        """
        nodes = truth.get_nodes()
        nodes_name = [node.get_name() for node in nodes]

        truePositives = np.zeros((len(nodes), len(nodes)))
        estPositives = np.zeros((len(nodes), len(nodes)))
        truePositivesCE = np.zeros((len(nodes), len(nodes)))
        estPositivesCE = np.zeros((len(nodes), len(nodes)))

        # Assumes the list of nodes for the two graphs are the same.
        for i in list(range(0, len(nodes))):
            for j in list(range(0, len(nodes))):
                if truth.get_endpoint(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])) == endpoint:
                    truePositives[j][i] = 1
                if est.get_endpoint(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])) == endpoint:
                    estPositives[j][i] = 1
                if truth.get_endpoint(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])) == endpoint \
                        and est.is_adjacent_to(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])):
                    truePositivesCE[j][i] = 1
                if est.get_endpoint(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])) == endpoint \
                        and truth.is_adjacent_to(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])):
                    estPositivesCE[j][i] = 1

        ones = np.ones((len(nodes), len(nodes)))
        zeros = np.zeros((len(nodes), len(nodes)))
        
        print(estPositives)
        print(truePositives)
        
        print(truePositives == estPositives)

        self.__Fp = (np.maximum(estPositives - truePositives, zeros)).sum()
        self.__Fn = (np.maximum(truePositives - estPositives, zeros)).sum()
        self.__Tp = (np.minimum(truePositives == estPositives, truePositives)).sum()
        self.__Tn = (truePositives == estPositives).sum() - self.__Tp

        self.__FpCE = (np.maximum(estPositivesCE - truePositivesCE, zeros)).sum()
        self.__FnCE = (np.maximum(truePositivesCE - estPositivesCE, zeros)).sum()
        self.__TpCE = (np.minimum(truePositivesCE == estPositivesCE, truePositivesCE)).sum()
        self.__TnCE = (truePositivesCE == estPositivesCE).sum() - self.__TpCE

    def get__fp(self):
        return self.__Fp

    def get__fn(self):
        return self.__Fn

    def get__tp(self):
        return self.__Tp

    def get__tn(self):
        return self.__Tn

    def get__fp_ce(self):
        return self.__FpCE

    def get__fn_ce(self):
        return self.__FnCE

    def get__tp_ce(self):
        return self.__TpCE

    def get__tn_ce(self):
        return self.__TnCE

    def get__precision(self):
        return self.__Tp / (self.__Tp + self.__Fp)

    def get__recall(self):
        return self.__Tp / (self.__Tp + self.__Fn)

    def get__precision_ce(self):
        return self.__TpCE / (self.__TpCE + self.__FpCE)

    def get__recall_ce(self):
        return self.__TpCE / (self.__TpCE + self.__FnCE)
    
    def get__F1(self):
        P = self.get__precision()
        R = self.get__recall()
        
        return (P*R)/(P+R)
    
    
    def get__F1_ce(self):
        P = self.get__precision_ce()
        R = self.get__recall_ce()
        
        return (P*R)/(P+R)
        
