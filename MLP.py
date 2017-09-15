import theano as th
import theano.tensor as T
import numpy as np
from compression import compressbatch as cb
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
def floatX(X):
    return np.asarray(X, dtype=th.config.floatX)

def weights(size):
	return th.shared(floatX(np.random.randn(*size)*0.01))
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
X=T.fmatrix('X')
Y=T.fmatrix('Y')
def network(w1,w2,X):
	hidden=T.dot(X,w1)
	out=T.dot(hidden,w2)
	out=softmax(out)
	return out
w1=weights((784,1024))
w2=weights((1024,10))
output=network(w1,w2,X)
output1=T.argmax(output,axis=1)
cost=T.mean(T.nnet.categorical_crossentropy(output, Y))
lr=0.1
updates=[(w1,w1-0.1*T.grad(cost,w1)),(w2,w2-0.1*T.grad(cost,w2))]
func=th.function([X,Y],cost,updates=updates,allow_input_downcast=True)
predict = th.function(inputs=[X], outputs=output1, allow_input_downcast=True)
print "building data trX"
trX=np.asarray(cb(mnist.train.images),dtype=th.config.floatX)
print trX.shape
print "building data teX"
teX=np.asarray(cb(mnist.test.images),dtype=th.config.floatX)
print teX.shape
print "collecting labels"
trY=np.asarray(mnist.train.labels,dtype=th.config.floatX)
print "collecting lables" 
teY=np.asarray(mnist.test.labels,dtype=th.config.floatX)
for i in range(10):
	print i
	for i in range(0,len(trX),100):
		print i,i+100
		cost = func(trX[i:i+100],trY[i:i+100])
		print cost
	print np.mean(np.argmax(teY, axis=1)==predict(teX))