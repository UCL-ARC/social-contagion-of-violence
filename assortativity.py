import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph()
g.add_nodes_from([0,1 ,3,2, ], size=1)
g.add_nodes_from([ 4], size=2)
g.add_edges_from([(0,1),  (4,0), (0,2), (0,3), ])
print(nx.numeric_assortativity_coefficient(g, 'size'))
print(nx.attribute_mixing_dict(g, 'size'))
print(nx.numeric_mixing_matrix(g,'size'))

pos = nx.spring_layout(g)
# nx.draw(g, pos=pos, ax=ax1, node_size=20, alpha=0.2)
color_map = {1:'blue',2:'red'}
nx.draw_networkx_nodes(g, pos, node_size=20, node_color=[color_map[k] for i, k in g.nodes.data('size')])
nx.draw_networkx_edges(g, pos, alpha=0.4, )
nx.draw_networkx_labels(g,  pos=pos,  font_size=20)
plt.show()
