import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import operator
from datetime import datetime, timedelta
try:
    import geopandas as gp #might need to install
    from shapely.geometry import Point
except: #If there is an exception in importing geopandas or shapely, we will ignore it and
        #simply not import the package.
    pass
from collections import defaultdict
from scipy.cluster import hierarchy
from scipy.spatial import distance

def get_unique_column_values(df,colname):
	return df[colname].unique()

def generate_directed_network(edges,labels):
    g=nx.DiGraph()
    edge_labels=dict(zip(edges,labels))
    for edge,label in zip(edges,labels):
        g.add_edge(edge[0],edge[1],label=label)
    return g,edge_labels

def generate_line_graph(edge_labels,graph):
    g2=nx.DiGraph()
    ln_graph=nx.line_graph(graph)
    for edge in ln_graph.edges():
        g2.add_edge(edge_labels[edge[0]],edge_labels[edge[1]])
    return g2

def generate_clustering_coefficient_plot(g):
    sns.set_style('whitegrid')
    #Ignore nodes with clustering coefficients of zero.
    clustering_coefficients=list(filter
        (lambda y: y[1]>0,sorted(
            nx.clustering(g).items(),key=lambda x: x[1],reverse=True)))
    plt.figure(figsize=(7,7))
    plt.plot(list(map(lambda x: x[1],clustering_coefficients)))
    plt.ylabel("Clustering Coefficient",fontsize=16)
    plt.xlabel("Number of Nodes",fontsize=16)

def get_indegree_and_outdegree(graph):
    """ 
        Return Indegrees
    """
    node_indegrees=list(graph.in_degree().items())
    node_outdegrees=list(graph.out_degree().items())
    return node_indegrees,node_outdegrees

def sort_by_degree(degrees_list,reverse=False):
    return sorted(degrees_list,key=operator.itemgetter(1),reverse=reverse)

def generate_degree_rank_plots(edges_with_weights):
    g=nx.Graph() #Instantiate an Undirected Graph.
    #Add all edges to DiGraph degardless of weight threshold.
    for edge_wt in edges_with_weights:
        g.add_edge(edge_wt['edge'][0],edge_wt['edge'][1])
    deg=list(sorted(nx.degree(g).values(),reverse=True)) #Consider all nodes.
    deg1=deg[10:-10] #Disergard ten nodes with the highest and least degree values.
    deg2=deg[20:-20] #Disregard twenty nodes with the highest and least degree values.
    
    fig,ax=plt.subplots(1,3,figsize=(20,5))
    ax[0].loglog(deg,'b-',marker='o')
    ax[1].loglog(deg1,'b-',marker='o',c='r')
    ax[2].loglog(deg2,'b-',marker='o',c='g')
    ax[0].set_ylabel('Degree',fontsize=18)
    ax[0].set_xlabel('Rank All',fontsize=18)
    ax[1].set_xlabel('Rank excluding \ntop and bottom 10 nodes',fontsize=18)
    ax[2].set_xlabel('Rank excluding \ntop and bottom 20 nodes',fontsize=18)

def generate_degree_rank_plot(edges_with_weights):
    g=nx.Graph() #Instantiate an Undirected Graph.
    #Add all edges to DiGraph degardless of weight threshold.
    for edge_wt in edges_with_weights:
        g.add_edge(edge_wt['edge'][0],edge_wt['edge'][1])
    
    deg=list(sorted(nx.degree(g).values(),reverse=True)) 
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    ax.loglog(deg,'b-',marker='o')
    ax.set_ylabel('Degree',fontsize=18)
    ax.set_xlabel('Rank',fontsize=18)

def load_citibike_data(inputfile,compression='gzip',headerrow=0,separator=','):
    """
        @param : inputfile: full path to the input file.
        @param : compression :  {‘infer’, ‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}
        @param : headerrow: Row number(s) to use as the column names (default 'infer')
        @param : separator: The symbol used to separate successive values in the file.
        
        Function reads in a csv file in any of the aforementioned `compression` formats and returns a "DataFrame".
    """
    df = pd.read_csv(inputfile,compression=compression,header=headerrow, sep=separator) 
    return df

def calculate_trip_durations_citibike(df):
    
    #convert the Start and End Time columns to datetime.
    df['Start Time']=pd.to_datetime(df['Start Time'])
    df['Stop Time']=pd.to_datetime(df['Stop Time'])

    #Trip Duration is End - Start time. This will result in datetime.timedelta objects, stored in the 'Trip Duration' column.
    df['Trip Duration']=df['Stop Time'] - df['Start Time']  #This is still timedelta.

    #Convert datetime.timedelta object Trip Duration to floating point number of minutes for ease of plotting.
    df['Trip Duration Minutes']=df['Trip Duration'].apply(lambda x: timedelta.total_seconds(x)/60.0)
    return df

def create_subset_graph(edges_with_weights,thr=0.005,graphtype='Directed'):
    """
        Creates a directed graph 
    """
    #edges[:len(weights)*0.3]
    if graphtype=='Directed':
        g=nx.DiGraph() #Instantiate a Directed Graph Object from NetworkX.
    elif graphtype=='UnDirected':
        g=nx.Graph() #Instantiate an Undirected Graph Object from NetworkX.
    
    thr=thr #Get top 0.5% edges by weight.
    edges_with_weights_new=list(filter(lambda edg: edg['edge'][0]!=edg['edge'][1],
                                   sorted(edges_with_weights,reverse=True,
                                          key=operator.itemgetter('weight'))))[:int(len(edges_with_weights)*thr)]

    #[:len(edges_with_weights)*0.1]
    for edge_wt in edges_with_weights_new:
        g.add_edge(edge_wt['edge'][0],edge_wt['edge'][1],weight=edge_wt['weight'])
    
    return g


def infer_weighted_station_station_network(df):
    dir_edges=dict()
    seen=list()
    for row in df.iterrows():
        st=row[1]['Start Station ID']
        end=row[1]['End Station ID']
        tripduration=row[1]['Trip Duration Minutes']
        if not st in seen:
            seen.append(st)
            dir_edges[st]={end:tripduration}
        else:
            try:
                dir_edges[st][end]+=tripduration
            except KeyError:
                dir_edges[st][end]=tripduration 
    
    edges_with_weights=list()
    #weights=list()
    for st_stn in seen:  
        end_stns=list(dir_edges[st_stn].keys())   
        num_end_stations=len(end_stns)
        for end_stn in end_stns:
            norm_wt=dir_edges[st_stn][end_stn]/num_end_stations
            edges_with_weights.append({'edge':(st_stn,end_stn),'weight':norm_wt})
            
    return edges_with_weights

def create_geodf_citibike_nyc(df,station_ids):
    
    geo_df_dict={'geometry':list(),'station_ids':list()}
    for stn_id in station_ids:  #Iterate over all station_ids.
        _df=df[df['Start Station ID']==stn_id]  #Filter rows where Start Station ID equals stn_id .
        if _df.shape[0]>0:
            lat=_df['Start Station Latitude'].values[0]  #Get the lat value of the particular station.
            lon=_df['Start Station Longitude'].values[0] #Get the lon value of the particular station.
            geo_df_dict['geometry'].append(Point(lon,lat))  #Add this as a Shapely.Point value under the 'geometry' key.
            geo_df_dict['station_ids'].append(stn_id)
        else: 
            _df=df[df['End Station ID']==stn_id]
            if _df.shape[0]>0:
                lat=_df['End Station Latitude'].values[0]
                lon=_df['End Station Longitude'].values[0]
                geo_df_dict['geometry'].append(Point(lon,lat))
                geo_df_dict['station_ids'].append(stn_id)
                
    geo_df_dict['geometry']=list(geo_df_dict['geometry'])
    geo_stations=gp.GeoDataFrame(geo_df_dict)
    geo_stations.drop(geo_stations[geo_stations['geometry']==Point(0,0)].index,inplace=True)
    geo_stations.reset_index(inplace=True)
    geo_stations.to_crs = {'init': 'epsg:4326'}
    return geo_stations

def plot_network(g,node_dist, nodecolor='g',nodesize=1200,nodealpha=0.6,edgecolor='k',\
                 edgealpha=0.2,figsize=(9,6),title=None,titlefontsize=20,savefig=False,\
                 filename=None,bipartite=False,bipartite_colors=None,nodelabels=None,
                 edgelabels=None):
    #pos=nx.spring_layout(g,iterations=200)
    pos=nx.spring_layout(g,k=node_dist,iterations=300)
    nodes=g.nodes()
    edges=g.edges()
    plt.figure(figsize=figsize)
    
    nx.draw_networkx_edges(g,pos=pos,edge_color=edgecolor,alpha=edgealpha)
    #nx.draw_networkx_edges(g,pos=pos,edge_color=edgecolor,alpha=edgealpha)
    if bipartite and bipartite_colors!=None:
        bipartite_sets=nx.bipartite.sets(g)
        _nodecolor=[]
        for _set in bipartite_sets:
            _clr=bipartite_colors.pop()
            for node in _set:
                _nodecolor.append(_clr)

        nx.draw_networkx_nodes(g,pos=pos,node_color=_nodecolor,alpha=nodealpha,node_size=nodesize)
    else:
        nx.draw_networkx_nodes(g,pos=pos,node_color=nodecolor,alpha=nodealpha,node_size=nodesize)

    labels={}
    for idx,node in enumerate(g.nodes()):
        labels[node]=str(node)

    if nodelabels!=None:
        nx.draw_networkx_labels(g,pos,labels,font_size=16)
    if edgelabels!=None: #Assumed that it is a dict with edge tuple as the key and label as value.
        nx.draw_networkx_edge_labels(g,pos,edgelabels,font_size=12)
    plt.xticks([])
    plt.yticks([])
    if title!=None:
        plt.title(title,fontsize=titlefontsize)
    if savefig and filename!=None:
        plt.savefig(filename,dpi=300)

def create_hc(G):
    """Creates hierarchical cluster of graph G from distance matrix"""
    path_length=nx.all_pairs_shortest_path_length(G)
    distances=np.zeros((len(G),len(G)))
    for u,p in path_length.items():
        for v,d in p.items():
            distances[u][v]=d
    # Create hierarchical cluster
    Y=distance.squareform(distances)
    Z=hierarchy.complete(Y)  # Creates HC using farthest point linkage
    # This partition selection is arbitrary, for illustrive purposes
    membership=list(hierarchy.fcluster(Z,t=1.15))
    # Create collection of lists for blockmodel
    partition=defaultdict(list)
    for n,p in zip(list(range(len(G))),membership):
        partition[p].append(n)
    return list(partition.values())

def draw_hc(G, lvl):
    plt.close()
    plt.ion()
    fig=plt.figure(figsize=(9,6))
    plt.axis("off")
    G=nx.convert_node_labels_to_integers(G)
    partitions=create_hc(G)
    BM=nx.blockmodel(G,partitions)
    node_size=[BM.node[x]['nnodes']*10 for x in BM.nodes()]
    edge_width=[1 for (u,v,d) in BM.edges(data=True)]
    pos = nx.spring_layout(BM,iterations=200)
    nx.draw(BM,pos,node_size=node_size,width=edge_width,with_labels=False)
    plt.title('Agglomerative Hierarchical Clustering for Citibike Network, lvl {}'.format(lvl))
    plt.show()
    return BM, partitions

def dhc(Gc):
    H = Gc.copy()
    partitions = {}
    partition_graph = {}
    lvl = 1
    # adding the original graph as the first lvl
    sub_graphs = nx.connected_component_subgraphs(H)
    partitions.update({lvl:[k.nodes() for k in sub_graphs]})
    while(nx.number_connected_components(H)!=H.number_of_nodes()):
        eb = nx.edge_betweenness(H) # use edge betweenness to find the best edge to remove
        sorted_eb = sorted(eb.items(), reverse=True, key=operator.itemgetter(1))
        k,v = sorted_eb[0][0]
        ncc_before = nx.number_connected_components(H)
        H.remove_edge(k,v)
        ncc_after = nx.number_connected_components(H)
        if ncc_before == ncc_after: continue
        else:
            lvl +=1
            sub_graphs = nx.connected_component_subgraphs(H)
            partitions.update({lvl:[k.nodes() for k in sub_graphs]})
            partition_graph.update({lvl:H.copy()})

    return partition_graph, partitions

def plot_dhc(PG, part, labels, lvl, pos):
    fig = plt.figure(figsize=(9,6))
    BM=nx.blockmodel(PG, part)
    node_size=[BM.node[x]['nnodes']*10 for x in BM.nodes()]
    edge_width=[1 for (u,v,d) in BM.edges(data=True)]
    BM_pos = nx.spring_layout(BM,iterations=200)
    plt.axis("off")
    plt.title("Block Model Clusters at level: {}".format(lvl))
    nx.draw(BM, BM_pos,node_size=node_size,width=edge_width,with_labels=False)
    fig = plt.figure(figsize=(9,6))
    plt.axis("off")
    plt.title("Node Clusters at level: {}".format(lvl))
    nx.draw_networkx(PG, pos, node_size=45, cmap = plt.get_cmap("jet"), node_color=labels, with_labels = False)

def generate_gspan_citibike(df, station2id, directed = False, time = 6):
    start_time = pd.DatetimeIndex(['2017-01-01T00:00:00.000000000'])
    end_time = pd.DatetimeIndex(['2017-02-01T00:00:00.000000000'])
    next_time = start_time + timedelta(hours=time)

    fw = open('../datasets/citibike/gspan_{}_{}.data'.format('directed' if directed else 'undirected',time),'w')
    graph_number = 0
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    fw.write('t # {}\n'.format(graph_number))
    while next_time < end_time:
        fl = df[(df['Start Time'].values > start_time.values) & (df['Start Time'].values <= next_time.values)]
        for sts,ens in zip(fl['Start Station ID'].values,fl['End Station ID'].values):
            if (sts,ens) in g.edges():
                g[sts][ens]['weight'] = g[sts][ens]['weight']+1
            else:
                g.add_edge(sts,ens,weight=1)

        node2id = {station2id[v]:k for k,v in enumerate(g.nodes())}
        for i, node in enumerate(g.nodes()):
            fw.write('v {} {}\n'.format(i,station2id[node]))
        for i, info in enumerate(g.edges(data=True)):
            fw.write('e {} {} {}\n'.format(node2id[station2id[info[0]]],node2id[station2id[info[1]]],info[2]['weight']))
        start_time = next_time
        next_time = start_time + timedelta(hours=time)
        graph_number += 1
        fw.write('t # {}\n'.format(graph_number))
        g.clear()
    fw.close()