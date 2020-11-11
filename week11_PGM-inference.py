"""
Inference in graphical models by message passing.
Marcus Frean and Tony Vignaux.
Last updated 2020 by mf.
"""
from sys import stderr

import numpy as np

def listLessOne(alist,k):
    return (alist[0:k] + alist[k+1:len(alist)])


class Node:
    def __init__(self,name=''):
        self.name=name
            
    def __str__(self):
        return self.name

    def ding(self, toNode,outboundMsg, dingChain):
        # This simply alerts a neighbour (toNode) that it needs to
        # respond to a message (outboundMsg)
        # print ('node',self.name,'dings',toNode.name,'(',dingChain,') with',outboundMsg)
        if dingChain < 10:
            dingChain = dingChain + 1
            toNode.respondToMessage(self,outboundMsg, dingChain)
        
    def respondToMessage(self,fromNode,inboundMsg, dingChain):
        # A node responds to an inbound message from fromNode, by
        # going through its other neighbours, recalculating messages
        # to them, and letting them know (via ding).
        k=self.edges.index(fromNode) # looks up the index
                                     # corresponding to fromNode
        self.msg[k]=inboundMsg
        receivers=listLessOne(self.edges,k)
        for r in receivers:
            i=self.edges.index(r)
            newMsg=self.calcMessage(i)
            self.ding(r,newMsg,dingChain)

        
class VariableNode(Node):
    def __init__(self,name='',vlen=1):
        Node.__init__(self,name)
        self.vlen=vlen
        self.edges=[]
        self.msg=[1]*len(self.edges)
        self.observed=False
            
    def __str__(self):
        return self.name

    def display(self):
        print ('--------------------------------------------')
        print ('node %s ' % self.name)
        if self.observed:
            print ('has been observed')
        for i in range(len(self.edges)):
            print ('from node: ',self.edges[i],' msg:',self.msg[i])
        p = np.prod(np.vstack(self.msg),0) # elt-wise multiplication
        print ('posterior: ', p/np.sum(p))

    def calcMessage(self,i):
        newMsg = np.prod(np.vstack(listLessOne(self.msg,i)),0)
        
        return newMsg

    def initialDing(self):
        # VariableNodes that are terminal nodes need to ding their 
        # edge with a message consisting of all ones.
        print ('Initial ding from terminal node',self.name,'to',self.edges[0].name)
        dingChain = 0
        self.ding(self.edges[0], np.ones((1,self.vlen),float), dingChain)

class FactorNode(Node):
    def __init__(self,name='',edges=[],phi=[]):
        Node.__init__(self,name)
        self.edges=edges
        self.msg=[1]*len(self.edges)
        for i in range(len(self.edges)):
            neighbour=self.edges[i]
            self.msg[i] = np.ravel(np.ones((1,neighbour.vlen),float)) # i.e. vector of ones
            
            # update the neighbour's edges and msgs
            neighbour.edges.append(self)
            neighbour.msg.append(self.msg[i])
        self.phi=phi
        # Check that dimensions of phi match the vlen's of variables.
        for i in range(len(edges)):
            if not((self.edges[i]).vlen == self.phi.shape[i]):
                print('Ooops: shape of',self.name,'phi doesnt match its variables.')
                print('Shape[',i,'] is',self.phi.shape[i])
                print('vlen of edge',self.edges[i],'is',(self.edges[i]).vlen)
                # ........AND WE SHOULD QUIT HERE, WITH ERROR MESSAGE...
                stderr.write('There is a mismatch between size of phi and of a message')
                

    def calcMessage(self,i):
        # We need to be able to leave out one dimension at will.
        # First, we rotate phi around so it's got the i-th axis first,
        # followed by the others in ascending order:
        nPhiDims = len(self.phi.shape)
        axesorder = [i] + listLessOne(list(range(nPhiDims)),i)
        z=np.transpose(self.phi, axes=(axesorder))
        indices=list(range(len(self.msg)-1))
        # Go through the other messages and "integrate them out." Each
        # time a variable is summed out like this the dimensionality
        # of z (i.e. phi) goes down by one. We're summing out the
        # right-most dimension of z. We go through msg in reverse
        # order so that the length of the msg and the *rightmost*
        # dimension of z match.
        indices.reverse()
        othermsg = listLessOne(self.msg,i)
        for j in indices:
            y=othermsg[j]*z
            z=np.transpose(sum(np.transpose(y),0))
        return z

    def initialDing(self):
        # FactorNodes that are terminal nodes need to ding their (one)
        # edge with the "message" phi.
        print ('Initial ding from terminal node',self.name,'to',self.edges[0].name)
        dingChain = 0
        self.ding(self.edges[0], self.phi, dingChain)

    def __str__(self):
        return self.name

    def display(self):
        print ('--------------------------------------------')
        print (self.name)
        for i in range(len(self.edges)):
            print ('intray ',self.edges[i],' msg:',self.msg[i])
        print ('factor parameters are:\n',self.phi)
#        q = np.array(1)
#        for z in self.msg:
#            q = multiply.outer(q,z)
#        q = q*self.phi
#        q = q / sum(q,len(self.edges)-1) # this normalises over the LAST INDEX...
#        print ('revised phi would appear to be...\n',q)


class Observation(FactorNode):
    # Called if a variable is observed - obs is the resulting vector.
    # 1. Must ding all neighbours with obs.
    # 2. Must somehow disable incoming dings.
    # 3. The variable has to "know" its observed value.
    # ALL THESE will occur if self simply acquires a new terminal FactorNode,
    # with zeros-bar-one (which dings back self once).
    # For some reason it works okay here but not in burglar.py
    def __init__(self,observedNode,obs=[]):
        FactorNode.__init__(self,'OBS',[observedNode],obs)
        self.edges=[observedNode]
        self.initialDing()
        observedNode.observed = True # just for humans, not used algorithmically

    
if __name__ == '__main__':


    print ('Testing...')
    # The Hidden Markov Model network is now in hmm.py The burglar
    # alarm test network is now in burglar.py Our first undirected
    # network is now in undirected_test.py Burglar model is used for
    # testing. This is the classic belief net example: burglar,
    # earthquake, alarm.

    # Define the variables first.
    b = VariableNode('burglar',2)
    e = VariableNode('earthquake',2)
    a = VariableNode('alarm',2)
    # And now the FactorNodes, each with a phi matrix.
    B = FactorNode('B',[b],np.array([.2,.8]))
    E = FactorNode('E',[e],np.array([.4,.6]))
    A = FactorNode('A',[b,e,a],np.array([[[.9,.1],[.6,.4]],[[.3,.7],[.5,.5]]]))

    theVariableNodes = [b,e,a]
    theFactorNodes = [B,E,A]
    theNodes = theVariableNodes + theFactorNodes

    # Initialise all messages: every terminal node dings its neighbour.
    print ('############################# Initialising all messages')
    for i in theNodes:
        if len(i.edges) == 1: # ie. it has one edge so it's a terminal node.
            i.initialDing()

    # Now we can get down to business....
    for i in theVariableNodes:
        i.display()
    A.display()

    print ('############################# observe',b.name)
    Observation(b,np.array([0.0,1.0]))
    for i in theVariableNodes:
        i.display()  # Notice e unchanged, despite message from A!
        # This wouldn't happen in general MRF graph, so it must be due
        # to normalisation...

    print('############################# observe',a.name)
    Observation(a,np.array([0.0,1.0]))
    for i in theVariableNodes:
        i.display() # e is different: "explaining away"
    A.display()

