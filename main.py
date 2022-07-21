import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from model_tf_4 import GTN 
import pdb
import pickle
import argparse
from utils import f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    args = parser.parse_args()
    
    print(args)
    
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
        
      
    num_nodes = edges[0].shape[0]
    num_nodes = 1000
    
    #print('num_nodes', num_nodes)
    #B = (tf.convert_to_tensor(edge.todense(), dtype= tf.float32))
    #print('B-shape',B.shape)

    # A = Adjacency matrix 
    
    for i,edge in enumerate(edges): # i goesthrough numbers [0,1,2,3...] and edge through edges.
        
        if i ==0:
             A = tf.expand_dims(tf.convert_to_tensor(edge.todense()[0:1000,0:1000], dtype= tf.float32), -1)
        else:
             A = tf.concat((A,tf.expand_dims(tf.convert_to_tensor(edge.todense()[0:1000,0:1000], dtype= tf.float32), -1)), -1) 
    
    A = tf.concat((A, tf.expand_dims(tf.convert_to_tensor(tf.eye(num_nodes), dtype= tf.float32), -1) ), -1)
    
    #val
    num_nodes1 = 400
    for i,edge in enumerate(edges): # i goesthrough numbers [0,1,2,3...] and edge through edges.
        
        if i ==0:
             A1 = tf.expand_dims(tf.convert_to_tensor(edge.todense()[0:400,0:400], dtype= tf.float32), -1)
        else:
             A1 = tf.concat((A1,tf.expand_dims(tf.convert_to_tensor(edge.todense()[0:400,0:400], dtype= tf.float32), -1)), -1) 
    
    A1 = tf.concat((A1, tf.expand_dims(tf.convert_to_tensor(tf.eye(num_nodes1), dtype= tf.float32), -1) ), -1)
 
    #test
    num_nodes2 = 400
    for i,edge in enumerate(edges): # i goesthrough numbers [0,1,2,3...] and edge through edges.
        
        if i ==0:
             A2 = tf.expand_dims(tf.convert_to_tensor(edge.todense()[0:400,0:400], dtype= tf.float32), -1)
        else:
             A2 = tf.concat((A2,tf.expand_dims(tf.convert_to_tensor(edge.todense()[0:400,0:400], dtype= tf.float32), -1)), -1) 
    
    A2 = tf.concat((A2, tf.expand_dims(tf.convert_to_tensor(tf.eye(num_nodes2), dtype= tf.float32), -1) ), -1)
    
   
    #print('shape',A.shape)
    
    node_features0 = node_features[0:1000,0:334]
    #print('shape',node_features.shape)
    node_features0 = tf.convert_to_tensor(node_features0, dtype= tf.float32)
    
    
    train_node = tf.convert_to_tensor(np.array(labels[0])[:,0])
    train_target = tf.convert_to_tensor(np.array(labels[0])[:,1], dtype = np.float32)
    
    #val
    node_features_val = node_features[0:400,0:334]
    node_features_val = tf.convert_to_tensor(node_features_val, dtype= tf.float32)
    
    valid_node = tf.convert_to_tensor(np.array(labels[0])[0:400,0])
    valid_target = tf.convert_to_tensor(np.array(labels[0])[0:400,1], dtype = np.float32)
    
    #test
    node_features_test = node_features[0:400,0:334]
    node_features_test = tf.convert_to_tensor(node_features_test, dtype= tf.float32)
    
    test_node = tf.convert_to_tensor(np.array(labels[0])[0:400,0])
    test_target = tf.convert_to_tensor(np.array(labels[0])[0:400,1], dtype = np.float32)
    
    num_classes = 1 #tf.get_static_value(tf.reduce_max(train_target)) +1
    # num_classes = tf.math.maximum(train_target).item()+1
    
    #print('num_class:' , num_classes)
    
    #print('train_node:' , train_node.shape)
    #print('train_target:' , train_target.shape)
    #print('node_f:' , node_features.shape)
    
    final_f1 = 0
    
    for l in tf.range(1):

        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        
        print('\n\n num_l',num_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        loss_tracker = keras.metrics.Mean(name="loss_t")
        mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
       
        

        for i in range(epochs):

            print("\nStart of epoch %d" % (i,))

            with tf.GradientTape() as tape:
                
                print('train shape')
                print(node_features0.shape)
                print(train_node.shape)
                print(train_target.shape)
                print('\n\n')
                      
                loss,y_train,Ws = model(A, node_features0, train_node, train_target)
              
                train_f1 = tf.reduce_mean(f1_score(tf.math.argmax(y_train, 1), train_target, num_classes=num_classes)).cpu() 
                
                print('Train - Loss: {}, Macro_F1: {}'.format(loss.cpu().numpy(), train_f1))

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
             # Valid

            print('valid shape')
            print(node_features_val.shape)
            print(valid_node.shape)
            print(valid_target.shape)

            print('\n\n')

            val_loss, y_valid,_ = model(A1, node_features_val, valid_node, valid_target)
            #val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
            val_f1 = tf.reduce_mean(f1_score(tf.math.argmax(y_valid, 1), valid_target, num_classes=num_classes)).cpu()
            #print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
            print('Train - Loss: {}, Macro_F1: {}'.format(val_loss.cpu().numpy(), val_f1))

            print('test shape')
            print(test_node.shape)
            print(test_target.shape)
            print('\n\n')                  
                  
            test_loss, y_test,W = model(A2, node_features_test, test_node, test_target)
            #test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
            test_f1 = tf.reduce_mean(f1_score(tf.math.argmax(y_test, 1), test_target, num_classes=num_classes)).cpu()
            #print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
            print('Train - Loss: {}, Macro_F1: {}'.format(test_loss.cpu().numpy(), test_f1))
                
            if val_f1 > best_val_f1:
                #best_val_loss = val_loss.detach().cpu().numpy()
                best_val_loss = tf.stop_gradient(val_loss).cpu().numpy()
                #best_test_loss = test_loss.detach().cpu().numpy()
                best_test_loss = tf.stop_gradient(test_loss).cpu().numpy()
                #best_train_loss = loss.detach().cpu().numpy()
                best_train_loss = tf.stop_gradient(loss).cpu().numpy()
                
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1 

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        final_f1 += best_test_f1


# In[ ]:




