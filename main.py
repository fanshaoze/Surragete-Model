import json

from analysis.data_2_GCN import parse_data_on_topology
from analysis.topoGraph import TopoGraph
from analysis.parseEfficiencyVOut import parse_eff_and_vout_from_csv

base_dir = '5comp/'
data_json = base_dir + 'data.json'
analysis_csv = base_dir + 'analytic.csv'
output = '5comp.json'


if __name__ == '__main__':
    # generate matrices file
    matrices = parse_data_on_topology(data_json)

    # parse eff, vout
    effs, vouts = parse_eff_and_vout_from_csv(analysis_csv)

    # using eff keys. if a graph is invalid, it's not in eff.keys()
    data = {}
    for name in effs.keys():
        node_list = matrices[name]['node_list']
        adj_matrix = matrices[name]['matrix']

        eff = effs[name]
        vout = vouts[name]

        paths = TopoGraph(node_list=node_list, adj_matrix=adj_matrix).find_end_points_paths_as_str()
        data[name] = {'node_list': node_list,
                      'eff': eff,
                      'vout': vout,
                      'paths': paths}

    with open(output, 'w') as f:
        json.dump(data, f)
