from z3 import *
from SupportFunctions import json_parser,z3_float,make_graph, XML_generator
from GraphPreProcessing import GraphPreProcessor
from time import time as tm
import math
import xml.dom.minidom as md

def Z3_Monomod(instance, method):

    print('begin processing instance: ',instance)
    if method == 'O':
        print('Optimizer')
    elif method == 'S':
        print('Solver')

    start_generation = tm()

    ATRs,nodes,Speed,AgentRadius = json_parser(f'TestInstances/{instance}.json')

    graph = make_graph(nodes)

    GPP = GraphPreProcessor(graph, AgentRadius,Speed)
    GPP.annotate_graph_with_ctc()

    # z3 variables

    # objective function dummy variable
    Z = Real('Z')

    NumberOfActions = 12

    at = {
        (v,n,a):Bool(f'{v}_AT_{n}_before_{a}')
        for v in ATRs
        for n in nodes
        for a in range(NumberOfActions+1)
    }

    start = {
        (v, a): Real(f'start_{v}_{a}')
        for v in ATRs
        for a in range(NumberOfActions)
    }

    end = {
        (v, a): Real(f'end_{v}_{a}')
        for v in ATRs
        for a in range(NumberOfActions)
    }
    move = {
        (v,n,a):Bool(f'{v}_TO_{n}_after_{a}')
        for v in ATRs
        for n in nodes
        for a in range(NumberOfActions)
    }

    wait = {
        (v, a): Real(f'wait_{v}_{a}')
        for v in ATRs
        for a in range(NumberOfActions)
    }

    # z3 constraints

    #CONSTRAINTS RELATED TO THE ROBOTS MOVEMENTS

    initial_location = [
        at[v,n[0],0] for v,n in ATRs.items()
    ]

    start_times = [
        start[v,0] == 0 for v in ATRs
    ]

    final_wait = [
        wait[v, NumberOfActions - 1] == 10 for v in ATRs #todo why is this necessary?
    ]

    wait_domain = [
        wait[v,a] >= 0 for v in ATRs for a in range(NumberOfActions)
    ]

    no_move_last_action = [
        Not(move[v,n,NumberOfActions-1]) for v in ATRs for n in nodes
    ]

    # no_two_wait_before_goal = [
    #     Implies(
    #         move[v, n[1], a],
    #         And([
    #             Implies(
    #                 wait[v, a1] > 0,
    #                 wait[v, a1 + 1] == 0
    #             )
    #             for a1 in range(a-2)
    #             ])
    #     )
    #     for v,n in ATRs.items()
    #     for a in range(NumberOfActions - 1)
    # ]

    start_vs_end =[
        end[v,a] == start[v,a+1] for v in ATRs for a in range(NumberOfActions-1)
    ]

    either_wait_or_move = [
        If(
            PbEq([ ( move[v,n,a], 1 ) for n in nodes ],1),
            wait[v,a] == 0,
            wait[v,a] > 0
        )
        for v in ATRs
        for a in range(NumberOfActions)
    ]
    final_location = [
        PbEq([
            ( move[v,n[1],a], 1 ) for a in range(NumberOfActions)
        ],1)
        for v, n in ATRs.items()
    ]

    # once the final location is reached, ther agent does not move any more
    no_move_after_goal = [
        Implies(
            move[v,n[1],a],
            And([
                wait[v,a2] > 0 for a2 in range(a+1,NumberOfActions)
            ])
        )
        for v,n in ATRs.items()
        for a in range(NumberOfActions)
    ]

    no_ubiquity = [
        PbEq([
            (at[v,n,a],1) for n in nodes
        ],1)
        for v in ATRs
        for a in range(NumberOfActions)
    ]

    uni_to = [
        PbLe([
            (move[v,n,a],1) for n in nodes
        ],1)
        for v in ATRs
        for a in range(NumberOfActions)
    ]

    not_moving_vehicle = [
        Implies(
            And(
                at[v,n,a],
                And([
                    Not(move[v,n2,a]) for n2 in nodes if n != n2
                ])
            ),
            And([
                at[v,n,a+1],
                end[v,a] == start[v,a] + wait[v,a]
            ])
        )
        for v in ATRs
        for n in nodes
        for a in range(NumberOfActions)

    ]

    moving_vehicle = [
        Implies(
            And(
                at[v,n,a],
                move[v,n2,a]
            ),
            And([
                at[v,n2,a+1],
                wait[v,a] == 0,
                end[v,a] == start[v,a] + math.dist(
                    (nodes[n]['x'],nodes[n]['y']),
                    (nodes[n2]['x'],nodes[n2]['y'])
                )/Speed
            ])
        )
        for v in ATRs
        for n in nodes
        for n2 in nodes
        for a in range(NumberOfActions)
    ]

    not_to_the_same = [
        Implies(
            at[v,n,a],
            Not(move[v,n,a])
        )
        for v in ATRs
        for n in nodes
        for a in range(NumberOfActions)
    ]

    only_adjacent_nodes = [
        Implies(
            at[v,n,a],
            And([
                Not(move[v,n2,a])
                for n2 in nodes
                if n2 not in nodes[n]['next']
            ])
        )
        for v in ATRs
        for n in nodes
        for a in range(NumberOfActions)
    ]

    # CONSTRAINTS FOR CONFLICT FREE ROUTING

    # if two vehicles are on the opposite sides of an edge, and one is crossing the edge, the other cannot cross
    # it until the first one is done traversing.

    move_move_conflict =[
        Implies(
            And(
                at[v1,n1,a1],
                move[v1, n3, a1],
                at[v2,n2,a2],
                move[v2, n4, a2],

            ),
            Or(
                start[v2,a2] >= start[v1,a1] + interval[1],
                start[v2,a2] <= start[v1,a1] + interval[0]
            )
        )
        for v1 in ATRs
        for v2 in ATRs
        if v2 != v1
        for (n1,n3),value in GPP.E_eec.items()
        for (n2,n4),interval in value.items()
        for a1 in range(NumberOfActions)
        for a2 in range(NumberOfActions)
    ]

    move_wait_conflict = [
        Implies(
            And(
                at[v1, n1, a1],
                wait[v1, a1] > 0,
                at[v2, n2, a2],
                move[v2, n4, a2]

            ),
            Or(
                start[v1, a1] >= start[v2, a2] + interval[1],
                end[v1, a1] <= start[v2, a2] + interval[0]
            )
        )
        for v1 in ATRs
        for v2 in ATRs
        if v2 != v1
        for (n2, n4), value in GPP.V_vec.items()
        for n1, interval in value.items()
        for a1 in range(NumberOfActions)
        for a2 in range(NumberOfActions)
    ]

    ########## OBJECTIVE FUNCTION #########

    opti = [
        Z
        ==
        Sum([
            If(
                move[v, n[1], a],
                end[v,a],
                0
            )
            for v,n in ATRs.items()
            for a in range(NumberOfActions)
        ])
    ]

    generation_time = round(tm() - start_generation,2)
    # print('generation: ', generation_time)
    if True:
        start_solving = tm()

        if method == 'O':
            s = Optimize()
        else:
            s = Solver()

        # making z3 more verbose
        # set_option("verbose", 2)
        set_option(rational_to_decimal=True)
        set_option(precision=20)
        # s.set('timeout',10800*1000)

        s.add(
            initial_location +
            start_times +
            # no_two_wait_before_goal +
            start_vs_end +
            final_wait +
            either_wait_or_move +
            wait_domain +
            no_move_last_action +
            final_location +
            no_move_after_goal +
            no_ubiquity +
            uni_to +
            not_moving_vehicle +
            moving_vehicle +
            not_to_the_same +
            only_adjacent_nodes +
            move_move_conflict +
            move_wait_conflict +
            opti
        )

        solution = {}

        # minimize the traveled distance
        if method == 'O':
            s.minimize(Z)

        # printing functions
        feasibility = s.check()
        if feasibility == sat:
            m = s.model()
            optimum = m[Z]
            print('Make Span: ', optimum)

            solution.update({'Z':optimum})

            if True:
                for v in ATRs:
                    print(f'Agent {v}')
                    for a in range(NumberOfActions):
                        print(f'    Action {a} - start: {z3_float(m[start[v, a]])}, end: {z3_float(m[end[v, a]])}')
                        buff = None
                        for n in nodes:
                            if m[at[v,n,a]]==True:
                                print(f'        {v} at node {n}')
                                buff = n

                        if float(z3_float(m[wait[v,a]])) > 0:
                            print(f'        {v} wait:',z3_float(m[wait[v,a]]))
                        for n in nodes:
                            if m[move[v,n,a]]==True:
                                dist = math.dist((nodes[n]['x'],nodes[n]['y']),(nodes[buff]['x'],nodes[buff]['y']))
                                print(f'        move to: {n} (distance: {dist})')

            # print(wait['A1', NumberOfActions - 1], m[wait['A1', NumberOfActions - 1]])
            # for n in nodes:
            #     print(move['A1',n,NumberOfActions-1], m[move['A1',n,NumberOfActions-1]])

            solution.update({
                'agents':{
                    v:{
                        'duration':z3_float(m[end[v,NumberOfActions-1]]),
                        'sections':{
                            a:{
                                'start_i':nodes[[n for n in nodes if m[at[v,n,a]]==True][0]]['x'],
                                'start_j': nodes[[n for n in nodes if m[at[v, n, a]] == True][0]]['y'],

                                'goal_i': nodes[[n for n in nodes if m[at[v, n, a]] == True][0]]['x']
                                if z3_float(m[wait[v,a]]) > 0
                                else nodes[[n for n in nodes if m[move[v, n, a]] == True][0]]['x'],
                                'goal_j': nodes[[n for n in nodes if m[at[v, n, a]] == True][0]]['y']
                                if z3_float(m[wait[v,a]]) > 0
                                else nodes[[n for n in nodes if m[move[v, n, a]] == True][0]]['y'],

                                'duration': round(z3_float(m[end[v,a]])-z3_float(m[start[v,a]]),2)
                            }
                            for a in range(NumberOfActions)
                        }
                    }
                    for v in ATRs
                }
            })

        else:
            print('model is ', feasibility)
            optimum = 'None'

    else:
        raise ValueError('WRONG SOLUTION METHOD')
    solving_time = round(tm() - start_solving,2)
    print('solving time: ', solving_time)

    stats = {
        'feasibility':feasibility,
        'optimum':optimum,
        'gen time':generation_time,
        'solving time':solving_time
    }

    return stats,solution

if __name__ == '__main__':

        instance = 'PaperExample'
        method = 'O'
        print('**************')
        stats,solution = Z3_Monomod(instance, method)

        # for i in solution.items():
        #     print(i)

        ATRs,nodes,Speed,AgentRadius = json_parser(f'TestInstances/{instance}.json')
        xml = XML_generator(ATRs,solution)

        dom = md.parseString(xml)
        pretty_xml_as_string = dom.toprettyxml()
        # print(pretty_xml_as_string)

        with open('CCBS_animation/AlvinExample/ce_task_log_CCBS.xml', 'w+') as f:
            f.write(pretty_xml_as_string)

