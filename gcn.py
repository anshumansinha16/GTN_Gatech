#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from inits import glorot, zeros
import pdb

class GCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        #self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        #self.weight = Parameter(tf.tensor(in_channels, out_channels))
        
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(shape=(w_in, w_out)),trainable=True)
        
        #self.bias = nn.Parameter(torch.Tensor(w_out))
        b_init = tf.random_normal_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(w_out,)),trainable=True)
        
        
        if bias:
            #self.bias = Parameter(torch.Tensor(out_channels))
            b_init = tf.random_normal_initializer()
            self.bias = tf.Variable(initial_value=b_init(shape=(w_out,)),trainable=True)
        
        else:
            #self.register_parameter('bias', None)
            self.bias = tf.Variable

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            #edge_weight = torch.ones((edge_index.size(1), ),
            #                         dtype=dtype,
            #                         device=edge_index.device)
            edge_weight = tf.ones((edge_index.size(1), ))
            
        # edge_weight = edge_weight.view(-1) # norm.view(-1, 1) * x_j
        edge_weight = tf.reshape(edge_weight,[-1]) # tf.reshape(norm,[-1,1])
        
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        #loop_weight = torch.full((num_nodes, ),
        #                         1 if not improved else 2,
         #                        dtype=edge_weight.dtype,
         #                        device=edge_weight.device)
        loop_weight = tf.fill((num_nodes, ), 1 if not improved else 2)
        
        #edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        edge_weight = tf.concat((num_nodes, ), loop_weight, dim=0)

        row, col = edge_index
        
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        #deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt = (1/deg)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0      

        return edge_index, deg_inv_sqrt[col] * edge_weight


    def call(self, x, edge_index, edge_weight=None):
        """"""
        #x = torch.matmul(x, self.weight)
        x = tf.matmul(X, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        #return norm.view(-1, 1) * x_j
        return tf.reshape(norm,[-1,1])*x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# In[13]:


import torch
import tensorflow as tf
x = torch.randn(2, 3, 4)
x1 = x.view(-1)
x2 = tf.reshape(x,[-1])
print(x1.shape)
print(x2.shape)


# In[9]:


x2


# In[10]:


x1


# In[17]:


edge_index, edge_weight = remove_self_loops(5, torch.tensor([4,7]))


# In[ ]:


@staticmethod
def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),dtype=dtype,
                                device=edge_index.device)
        
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    

