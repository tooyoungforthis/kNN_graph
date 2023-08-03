import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List


def draw_graph(Graph_1: nx.Graph, labels_1: Dict[int, int],
               Graph_2: nx.Graph, labels_2: Dict[int, int]) -> None:
    """Этот метод визуализирует исходный и граф с доп. вершиной
    с кластеризацией и без

    Args:
        Graph_1 (nx.Graph): Исходный граф
        label_1 (Dict[int, int]): Метки Graph_1
        Graph_2 (nx.Graph): Граф с доп. вершиной
        label_2 (Dict[int, int]): Метки Graph_3
    """

    # Сделаем одинаковое расположение вершин при визуализации
    pos_graph_1 = nx.spring_layout(Graph_1)
    pos_graph_2 = nx.spring_layout(Graph_2)
    for k, v in pos_graph_1.items():
        pos_graph_2[k] = v

    color_map_1 = _get_color_map(Graph_1, labels_1)
    color_map_2 = _get_color_map(Graph_2, labels_2)

    fig, axs = plt.subplots(2, 2)
    nx.draw(Graph_1, with_labels=True, pos=pos_graph_1, ax=axs[0, 0])
    nx.draw(Graph_1, with_labels=True,
            node_color=color_map_1, pos=pos_graph_1, ax=axs[1, 0])
    nx.draw(Graph_2, with_labels=True, pos=pos_graph_2, ax=axs[0, 1])
    nx.draw(Graph_2, with_labels=True,
            node_color=color_map_2, pos=pos_graph_2, ax=axs[1, 1])

    axs[0, 0].set_title(
        f"Исходный граф с количеством вершин = {len(Graph_1.nodes())}")
    axs[1, 0].set_title(
        f"Кластеризированный граф с количеством вершин = {len(Graph_1.nodes())}")
    axs[0, 1].set_title(
        f"Граф с дополнительно добавленной вершиной ({len(Graph_2.nodes())})")
    axs[1, 1].set_title(
        f"Граф с найденным классом для добавленной вершины"
    )

    plt.show()


def _get_color_map(Graph: nx.Graph, labels: Dict[int, int]) -> List[str]:
    """Этот метод возвращает цвета вершин графа

    Args:
        Graph (nx.Graph): Граф
        labels (Dict[int, int]): Метки вершин

    Returns:
        List[str]: Цвета вершин

    """
    color_map = {}
    for key in labels:
        if labels[key] == 0:
            color_map[key] = 'red'
        elif labels[key] == 1:
            color_map[key] = 'blue'
        else:
            color_map[key] = 'green'

    clusters_colors = [color_map.get(node) for node in Graph.nodes()]
    return clusters_colors
