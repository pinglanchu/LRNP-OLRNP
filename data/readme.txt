These datasets rperesent the adjacency matrix of networks. note that these datasets contain self-loop. For example, 

WS = nx.watts_strogatz_graph(1000, 10, 0.2, seed=0)
WS_adj = np.array(nx.adjacency_matrix(WS).todense()) + np.eye(1000)
np.save('WS.npy', WS_adj)

