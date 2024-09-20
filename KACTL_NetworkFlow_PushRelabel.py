LOCAL = 0
if LOCAL:
    import networkx as nx                              
    from matplotlib import pyplot as plt        



class Edge:
    def __init__(self,d,b,f,c):
        self.dest=d
        self.back=b
        self.f=f
        self.c=c
    def __repr__(self):
        return f'Edge({self.dest},{self.back},{self.f},{self.c})'

class PushRelabel:
    def __init__(self, n):
        self.g = [[] for _ in range(n)] # list of lists of Edges
        self.ec = [0]*n # list of ints
        self.cur = [None for _ in range(n)] # list of list of Edges
        self.curi = [0 for _ in range(n)] # list of list of Edges
        self.hs = [[] for _ in range(2*n)] # list of list of ints
        self.H = [0]*n # list of ints
        self.nedge = 0

    def addEdge(self, s, t, cap, rcap=0):
        if s==t: return
        self.g[s].append( Edge(t, len(self.g[t]), 0, cap) )
        self.g[t].append( Edge(s, len(self.g[s])-1, 0, rcap) )
        #:print(self.g[s][-1])
        self.nedge += 1

    def addFlow(self, e, f):
        back = self.g[e.dest][e.back]
        if (self.ec[e.dest]==0 and f>0): self.hs[self.H[e.dest]].append(e.dest)
        e.f += f; e.c -= f; self.ec[e.dest] += f
        back.f -= f; back.c += f; self.ec[back.dest] -= f
    
    def calc(self, s, t):
        v = len(self.g); self.H[s] = v; self.ec[t]=1
        co = [0]*(2*v); co[0] = v-1;
        for i in range(v):
            self.curi[i] = 0
            self.cur[i] = self.g[i][self.curi[i]]
        for e in self.g[s]:
            self.addFlow(e, e.c)
        hi = 0
        while True:
            while len(self.hs[hi])==0:
                if hi==0:
                    hi -= 1
                    return -self.ec[s]
                else:
                    hi -= 1
            u = self.hs[hi].pop()
            while self.ec[u] > 0:
                if self.curi[u] == len(self.g[u]):
                    self.H[u] = 1E9
                    for ui,e in enumerate(self.g[u]):
                        if e.c>0 and self.H[u]>self.H[e.dest]+1:
                            self.H[u] = self.H[e.dest]+1;
                            self.cur[u] = e
                            self.curi[u] = ui
                    co[self.H[u]] += 1; co[hi] -= 1
                    if co[hi]==0 and hi < v:
                        for i in range(v):
                            if hi < self.H[i] and self.H[i] < v:
                                co[self.H[i]] -= 1; self.H[i] = v+1
                    hi = self.H[u]
                elif self.cur[u].c>0 and self.H[u] == self.H[self.cur[u].dest]+1:
                    self.addFlow( self.cur[u], min(self.ec[u],self.cur[u].c) )
                else:
                    self.curi[u] += 1
                    self.cur[u] = self.g[u][self.curi[u]] if self.curi[u]<len(self.g[u]) else None

    def leftOfMinCut(self, a): return self.H[a] >= len(self.g)

    def retrieveFlow(self):
        edges = []
        for i in range(len(self.g)):
            for e in self.g[i]:
                if e.f>0:
                    edges.append( (i,e.dest,e.f) )
        return edges

    def retrieveCut(self):
        cut_edges = []
        for u,v,f in self.retrieveFlow():
            if self.H[u]>=len(self.g) and (not self.H[v]>=len(self.g)):
                cut_edges.append( (u,v,f) )
        return cut_edges

    def plotGraph(self):                               
        if LOCAL:                                      
            G = nx.DiGraph()                                                                                  
            for i, edges in enumerate(self.g):         
                for edge in edges:                     
                    j, capacity = edge.dest, edge.c                                                           
                    if capacity:                                                                              
                        G.add_edge(i, j, weight=capacity)
            # other layout options include: random, circular, spiral, spring, kamada_kawai, etc               
            layout = nx.shell_layout(G)
            nx.draw(G, layout, with_labels=True)
            nx.draw_networkx_edge_labels(G, layout, edge_labels={e: G.edges[e]["weight"] for e in G.edges})
            plt.show()                                                                                        




if __name__ == '__main__':

    N,M = map(int,input().split())

    SRC,DEST,NTERM = range(3)
    NODE_BASE = NTERM
    TOT_NODES = NODE_BASE+N
    G = PushRelabel(TOT_NODES)

    
    G.addEdge(SRC,TBD,1)
    for _ in range(M):
        u,v,c = map(int,input().split())
        G.addEdge(u,v,c)
    G.addEdge(TBD,DEST,1)

    flow = G.calc(SRC,DEST)
    print(flow)

    '''
    cut = G.retrieveCut()
    all_edges = G.retrieveFlow()
    verts_left = [u for u in range(N)
    '''
