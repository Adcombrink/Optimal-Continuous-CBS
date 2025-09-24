import json
import networkx as nx
from itertools import islice
import csv
import math
import lxml.etree
import lxml.builder


# this function takes a list of edges (supposedly belonging to a graph) and calculates
# the distance between any two points in the graph when taking the given path
def distance(path,edges):
    total_distance = 0
    for i in range(len(path)-1):
        # print(path[i],path[i + 1],edges[(path[i] , path[i+1])])
        total_distance += edges[(path[i] , path[i+1])][0]
    return total_distance

# builds a graph structure out of nodes and edges
def make_graph(nodes):

    graph = nx.DiGraph()

    for name,node in nodes.items():
        graph.add_node( name, pos = (node['x'], node['y']) )

    for name,node in nodes.items():
        for other_node in node['next']:
            graph.add_edge(name,other_node, weight = math.dist(
                                                   (nodes[name]['x'],nodes[name]['y']),
                                                   (nodes[other_node]['x'],nodes[other_node]['y'])
                                                   ))
    return graph

def json_parser(file_to_parse):
    with open(file_to_parse,'r') as read_file:
        data = json.load(read_file)

    nodes = data['test_data']['nodes']
    ATRs = data['ATRs']
    Speed = data['test_data']['Speed']
    AgentRadius = data['test_data']['AgentRadius']

    return ATRs,nodes,Speed,AgentRadius

def z3_float(input):
    toString = str(input)

    if toString[-1] == '?':
        return float(toString[:-1])
    elif toString == 'None':
        return None
    else:
        return float(toString)

def XML_generator(ATRs,solution):

    makespan = solution['Z']
    time = 0 #TODO
    flowtime = 1984 #TODO
    NumberOfActions = 4

    E = lxml.builder.ElementMaker()
    ROOT = E.root
    the_doc = ROOT()

    AGENT = E.agent

    SECTION = E.section
    PATH = E.path

    LOG = E.log
    the_log = LOG()

    for i, j in ATRs.values():
        the_doc.append(
            AGENT(start_id=i[1],goal_id=j[1])
        )


    SUMMARY = E.summary
    the_log.append(
        SUMMARY(time=str(time),flowtime=str(flowtime),makespan=str(makespan))
    )

    for v in ATRs:
        the_agent = AGENT(number=v[1])
        the_path = PATH(duration=str(solution['agents'][v]['duration']))
        for a in range(NumberOfActions):

            the_path.append(
                SECTION(
                    number=str(a),
                    start_i = str(solution['agents'][v]['sections'][a]['start_i']),
                    start_j = str(solution['agents'][v]['sections'][a]['start_j']),
                    goal_i = str(solution['agents'][v]['sections'][a]['goal_i']),
                    goal_j = str(solution['agents'][v]['sections'][a]['goal_j']),
                    duration = str(solution['agents'][v]['sections'][a]['duration'])
                )
            )

        the_agent.append(
            the_path
        )
        the_log.append(the_agent)

    the_doc.append( the_log )

    xml_string = lxml.etree.tostring(the_doc)
    return xml_string
