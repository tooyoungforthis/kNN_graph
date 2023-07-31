from graph_with_clusters_generator import generate_clustered_graph, clustering_graph, draw_graph
from collections import Counter
from typing import Dict, List
from itertools import combinations
import networkx as nx


class Graph_KNearestNeighbor:
    """kNN классификатор"""

    def __init__(self) -> None:
        pass

    def fit(self, node_and_its_cluster: Dict[int, int]):
        """
        Обучение классификатора
        Для kNN - это запоминание входных данных

        Входные данные:
        node_and_its_cluster - представление графа в виде: вершина - кластер
        """
        self.data = node_and_its_cluster

    def predict(self, X: List[int], k=1) -> int:
        """
        Предсказываем кластер, используя данный классификатор

        Входные данные:
        X - список вершин, с которыми вершина образует ребра 
        k - количество соседей, для выбора кластера

        Выходные данные: 
        y - предсказанный кластер для вершины, образующей ребра с вершинами X 

        Если для двух классов мы имеем одинаковые вероятности, 
        то будет использовать меньший по индексу
        """
        probabilities = self.compute_probabilities(X, k=k)
        predicted_cluster = probabilities.index(max(probabilities))

        return predicted_cluster

    def compute_probabilities(self, X: List[int], k=1) -> List[int]:
        """
        Будем перебирать C(X, k) комбинаций возвомжных перестановок соседей
        для поиска наиболее ВЕРОЯТНОГО кластера
        Вероятностью будем считать отношение кол-ва ребер n-го кластера в комбинации на общее кол-во ребер в n-ом кластере 
        Такая вероятность более чувствительна к миноритарным кластерам
        """
        clusters_count = Counter(self.data.values())
        max_probabiltity = [0] * len(clusters_count)

        for combination in combinations(X, k):
            X_nodes_clusters = [self.data[x] for x in combination]
            X_nodes_clusters_counter = Counter(X_nodes_clusters)
            # print(clusters_count)
            # print(X_nodes_clusters_counter)

            for indx, value in enumerate(max_probabiltity):
                probability_indx = X_nodes_clusters_counter.get(
                    indx, 0) / clusters_count.get(indx)
                max_probabiltity[indx] = max(value, probability_indx)

        return max_probabiltity
