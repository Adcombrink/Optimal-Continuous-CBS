# Import the required modules
import xmltodict
import pprint
import networkx as nx
import matplotlib.pyplot as plt
import math

# Open the file and read the contents
with open('CCBS_animation/AlvinExample/ce_roadmap.xml', 'r', encoding='utf-8') as file:
    my_xml = file.read()

# Use xmltodict to parse and convert
# the XML document
xml_dict = xmltodict.parse(my_xml)

# Print the dictionary
# pprint.pprint(xml_dict, indent=4)

nodes = {
    elem['@id']:[
        float(elem['data']['#text'].split(',')[0]),
        float(elem['data']['#text'].split(',')[1])
    ]
    for elem in xml_dict['graphml']['graph']['node']
}
edges = [
    (elem['@source'],elem['@target']) for elem in xml_dict['graphml']['graph']['edge']
]
nodes_dict = {
        i:{
            'pos':nodes[i],
            'next':[j[1] for j in edges if j[0] == i]}
        for i in nodes
    }
G = nx.Graph()
for i in nodes_dict:
    G.add_node(i)
for e in edges:
    G.add_edge(e[0], e[1])
nx.set_node_attributes(G, nodes_dict)

for node, coord in nodes.items():
    print(f'\"{node}\":{{\"x\":{coord[0]},\"y\":{coord[1]},\"next\":{nodes_dict[node]["next"]}}},')

for i in edges:
    print(f'\"{i[0]},{i[1]}\":[{round(math.dist((nodes[i[0]]),(nodes[i[1]])))},1],')


fig, ax = plt.subplots()

nx.draw(G,
            nx.get_node_attributes(G,'pos'),
            ax=ax,
            # node_size = 10,
            # font_size = 6,
            with_labels=True)

plt.show()


