from Z3_CCBS import Z3_Monomod
from SupportFunctions import json_parser, XML_generator
import xml.dom.minidom as md
from CCBS_animation.solution_animation import load_config,graph_from_path, extract_solution, animate_solution


instance = 'PaperExample'
method = 'O'
print('**************')
stats,solution = Z3_Monomod(instance, method)

# for i in solution.items():
#     print(i)

ATRs, _, _, _ = json_parser(f'TestInstances/{instance}.json')
xml = XML_generator(ATRs,solution)

dom = md.parseString(xml)
pretty_xml_as_string = dom.toprettyxml()
# print(pretty_xml_as_string)

with open('CCBS_animation/AlvinExample/ce_task_log_CCBS.xml', 'w+') as f:
    f.write(pretty_xml_as_string)

# Paths
output_file = 'CCBS_animation/AlvinExample/Animation.mp4'
path_config = 'CCBS_animation/AlvinExample/ce_config.xml'
path_roadmap = 'CCBS_animation/AlvinExample/ce_roadmap.xml'
path_task = 'CCBS_animation/AlvinExample/ce_task.xml'
path_log = 'CCBS_animation/AlvinExample/ce_task_log_CCBS.xml'

# load data
config = load_config(path_config)['root']['algorithm']
graph = graph_from_path(path_roadmap)
log = extract_solution(path_log)

# animate
animate_solution(config, graph, log, save_path=output_file)

print('done!')