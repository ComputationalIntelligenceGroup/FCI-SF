# -*- coding: utf-8 -*-

from IncrementalGraph import IncrementalGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils    import GraphUtils
from EndpointConfusion import EndpointConfusion

g1 = IncrementalGraph(5)
nodes1 = g1.G.get_nodes()

g1.G.add_edge(Edge(nodes1[0], nodes1[2], Endpoint.CIRCLE, Endpoint.ARROW))
g1.G.add_edge(Edge(nodes1[2], nodes1[4], Endpoint.CIRCLE, Endpoint.ARROW))
g1.G.add_edge(Edge(nodes1[1], nodes1[3], Endpoint.CIRCLE, Endpoint.ARROW))
g1.G.add_edge(Edge(nodes1[3], nodes1[4], Endpoint.CIRCLE, Endpoint.ARROW))



GraphUtils.to_pydot(g1.G).write_png("trial-PAG1-EndpointConfusion.png")

g2 = IncrementalGraph(5)
nodes2 = g2.G.get_nodes()


g2.G.add_edge(Edge(nodes2[0], nodes2[1], Endpoint.ARROW, Endpoint.TAIL))
g2.G.add_edge(Edge(nodes2[2], nodes2[4], Endpoint.CIRCLE, Endpoint.ARROW))
g2.G.add_edge(Edge(nodes2[1], nodes2[3], Endpoint.CIRCLE, Endpoint.ARROW))
g2.G.add_edge(Edge(nodes2[3], nodes2[4], Endpoint.CIRCLE, Endpoint.ARROW))

GraphUtils.to_pydot(g2.G).write_png("trial-PAG2-EndpointConfusion.png")




end_conf = EndpointConfusion(g1.G, g2.G, Endpoint.ARROW)




