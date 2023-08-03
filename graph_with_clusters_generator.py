import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Set, List, Mapping
from statistics import mode


def generate_clustered_graph(N: int, n_clusrers: int) -> nx.Graph:
    """Этот метод генерирует граф c N вершинами c разбиением на n_clusters кластеров
    c использованием gaussian_random_partition_graph

    Args:
        N (int): Количество вершин графа
        n_clusters (int): Количество кластеров

    Returns:
        nx.Graph: Граф с N вершинами c разбиением на n_clusters
    """
    # Зададим параметры для генерации
    mean_cluster_size = N // n_clusrers
    v = N * 10000
    p_in = 0.99
    p_out = n_clusrers / N
    seed = 42

    Graph = nx.gaussian_random_partition_graph(
        n=N, s=mean_cluster_size, v=v, p_in=p_in, p_out=p_out, seed=seed)

    return Graph


def clustering_graph(Graph: nx.Graph, n_clusters: int) -> Dict[int, int]:
    """Этот метод кластеризует граф на основе расстояния до ближайщих соседей

    Args:
        Graph (nx.Graph): Граф, который нужно кластеризировать
        n_clusters (int): Количество кластеров

    Returns:
        Dict[int, int]: Ключ - номер вершины, значение - номер кластера
    """
    shortest_path_matrix = _get_shortest_path_matrix(Graph)
    nodes_neighbours = _get_closest_neighbours(Graph, shortest_path_matrix)

    clusters = _compute_cluster_(Graph, n_clusters, nodes_neighbours)

    return clusters


def _get_shortest_path_matrix(Graph: nx.Graph) -> np.ndarray:
    """Этот метод рассчитывает кратчайшее расстояние между всеми парами вершин

    Args:
        Graph (nx.Graph): Исходный граф

    Returns:
        np.ndarray: Матрица кратчайших расстояний между всеми парами вершин
    """
    path = np.array(list(nx.all_pairs_dijkstra_path_length(Graph)))

    shortest_path_matrix = np.array([])
    for i in range(len(path)):
        shortest_path_matrix = np.append(shortest_path_matrix, np.array(
            [j[1] for j in sorted(path[i][1].items())]))

    try:
        shortest_path_matrix = shortest_path_matrix.reshape(
            len(Graph.nodes()), -1)
    except ValueError:
        print("Сгенерировался несвязный граф, увеличить p_out")

    return shortest_path_matrix


def _get_closest_neighbours(Graph: nx.Graph,
                            shortest_path_matrix: np.ndarray) -> Dict[int, Set[int]]:
    """Этот метод находит соседей каждой вершины
        Т.е вершины, которые образуют с ней ребра
        Также включаем саму вершину

        Args:
            Graph (nx.Graph): Исходный граф
            shortest_path_matrix (nd.ndarray): Матрица кратчайших путей

        Returns:
            Dict[int, Set[int]]: Соседи каждой вершины включительно с вершиной
        """
    nodes_neighbours: dict = {i: set() for i in Graph.nodes()}

    for node, nodes_distances in enumerate(shortest_path_matrix):
        for indx, value in enumerate(nodes_distances):
            if value <= 1:
                nodes_neighbours[node].add(indx)

    return nodes_neighbours


def _compute_cluster_(Graph: nx.Graph, n_clusters: int,
                      nodes_neighbours: Dict[int, Set[int]]) -> Dict[int, int]:
    """Этот метод находит кластеры для большинства вершин

    Args:
        Graph (nx.Graph): Исходный граф
        n_clusters (int): Количество кластеров
        nodes_neighbours (Dict[int, Set[int]]): Соседи каждой из вершин

    Returns:
        Dict[int, int]: Ключ - номер вершины, значение - номер кластера
    """
    clusters: List[List[int]] = [[]] * n_clusters

    for i in list(Graph.nodes()):
        res = [i]
        for j in list(Graph.nodes())[i + 1:]:
            temp = nodes_neighbours[i] & nodes_neighbours[j]
            if len(temp) >= len(nodes_neighbours[j]) - int(len(nodes_neighbours[j]) // n_clusters):
                res.append(j)
            else:
                for k in range(len(clusters)):
                    if set(res) >= set(clusters[k]):
                        clusters[k] = res
                        break
                    elif set(clusters[k]) >= set(res):
                        break
                    elif clusters[k] == []:
                        clusters[k] = res
                        break

    node_and_its_cluster = {}
    for k in Graph.nodes():
        for indx, cluster in enumerate(clusters):
            for value in cluster:
                if value == k:
                    node_and_its_cluster[k] = indx
                    continue
                continue
            continue

    clustered_nodes = [node for cluster in clusters for node in cluster]
    need_to_cluster_nodes = sorted(
        list(set(Graph.nodes()) - set(clustered_nodes)))
    # print(need_to_cluster_nodes)
    if need_to_cluster_nodes == []:
        return node_and_its_cluster

    node_and_its_cluster = _compute_cluster_for_particular_nodes(
        Graph, node_and_its_cluster, need_to_cluster_nodes, nodes_neighbours)

    return node_and_its_cluster


def _compute_cluster_for_particular_nodes(
        Graph: nx.Graph, node_and_its_cluster: Dict[int, int], need_to_cluster_nodes: List[int], nodes_neighbours: Dict[int, Set[int]]) -> Dict[int, int]:
    """Этот метод находит кластеры точек,
    которые не смогли найти на предыдущем этапе
    Конечный кластер будет определяться,
    как самый часто соседствующий кластер(мода)
    по полученным кластерам ближайших соседей

    Args:
        Graph (nx.Graph): Исходный граф
        node_and_its_cluster (Dict[int, int]): Ключ - номер вершины, значение - номер кластера
        need_to_cluster_nodes (List[int]): Некластеризованные вершины
        nodes_neighbours (Dict[int, Set[int]]): Соседи каждой из вершин

    Returns:
        Dict[int, int]: Ключ - номер вершины, значение - номер кластера
    """
    # определим кластеры
    while need_to_cluster_nodes:
        for node in need_to_cluster_nodes:
            neighbours_clusters = []
            for i in nodes_neighbours[node]:
                value = node_and_its_cluster.get(i, None)
                if value is None:
                    continue
                else:
                    neighbours_clusters.append(value)
            # print(node, neighbours_clusters)
            try:
                node_and_its_cluster[node] = mode(neighbours_clusters)
            except Exception as e:
                print(e)
            else:
                need_to_cluster_nodes.remove(node)

    return node_and_its_cluster


def draw_graph(Graph: nx.Graph, clusters: Dict[int, int], pos) -> None:
    """Этот метод выводит исходный граф и кластеризированный граф

    Args:
        Graph (nx.Graph): Исходный граф
        clusters (Dict[int, int]): Ключ - номер вершины, значение - номер кластера
        pos : Координаты каждой вершины
    """
    plt.figure(1)
    nx.draw(Graph, with_labels=True, pos=pos)

    plt.figure(2)
    color_map = {}
    for key in clusters:
        if clusters[key] == 0:
            color_map[key] = 'red'
        elif clusters[key] == 1:
            color_map[key] = 'blue'
        else:
            color_map[key] = 'green'

    clusters_colors = [color_map.get(node) for node in Graph.nodes()]
    nx.draw(Graph, with_labels=True, node_color=clusters_colors, pos=pos)

    plt.show()
