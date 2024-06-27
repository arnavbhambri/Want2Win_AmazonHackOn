#hotspot
import pandas as pd
from pyvis.network import Network
import networkx as nx

def amplified_expolog_pr(G, personalization=None, alpha=0.85, max_iter=100, tol=1e-6, transformation='exponential', amplify_factor=2.07):
    total = sum(personalization.values())
    personalization = {k: v / total for k, v in personalization.items()}
    #Real Gs normalize the weight    
    N = G.number_of_nodes()
    x = {n: 1.0 / N for n in G.nodes()}
    p = personalization

    #Hold those who cannot connect.
    dangling_weights = p
    dangling_nodes = [n for n in G if G.out_degree(n, weight='weight') == 0.0]

    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in G[n]:
                x[nbr] += alpha * xlast[n] / G.out_degree(n)
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
        if transformation == 'exponential':
            x = {k: np.exp(v ** amplify_factor) for k, v in x.items()}
        elif transformation == 'logarithmic':
            x = {k: np.log1p(v * amplify_factor) for k, v in x.items()}
        norm = sum(x.values())
        #aakhri baar normalize
        x = {k: v / norm for k, v in x.items()}
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x

    raise nx.PowerIterationFailedConvergence(max_iter)


def getnet() -> Network:
    df = pd.read_csv("csv/hotspots.csv")
    df_desc = pd.read_csv("csv/location_database.csv")
    G = nx.Graph()  

    for idx, row in df.iterrows():
        G.add_edge(row['H1'], row['H2'], weight=row['Weight'])

    net = Network(notebook=True)
    weighted_degrees = dict(G.degree(weight='weight'))
    pr_ranks = nx.pagerank(G, personalization=weighted_degrees)
    print(pr_ranks)
    top_list = sorted(pr_ranks.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10]
    finals = {}
    for item in top_list:
        location = df_desc.loc[df_desc['Warehouse_ID'] == item[0], 'Warehouse_Location'].values[0]
        finals[item[0]] = (location, item[1])

    check_list = [item for item in finals]
    net = Network(notebook=True)  

    for node in G.nodes():
        if(node in check_list):
            print(node)
            net.add_node(node, label=node, size=(pr_ranks[node]*297)**1.88, color='#f80000', title= finals[node])
        else:
            net.add_node(node, label=node, size=(pr_ranks[node]*297)**1.88, color='#FFA500')  

    for edge in G.edges(data=True):
        if(edge[2]['weight'] > 50):
            net.add_edge(edge[0], edge[1], value=edge[2]['weight'], color='#808080')  
    net.set_options("""
    var options = {
  "nodes": {
    "font": {
      "size": 16,
      "align": "center"
    }
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "smooth": {
      "type": "continuous"
    }
  },
  "physics": {
    "minVelocity": 0.75
  }
}
""")
    return net