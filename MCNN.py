import theano as th
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.ifelse import ifelse
from compression import compressbatch as cb
from compression import reducebatch as rb
from compression import printimage as pi 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

srng = RandomStreams()
def inner_i(data):
	result,updates=th.scan(fn=lambda i:ifelse(T.gt(T.cast(i,'float64'),0.),1.,0.),
							outputs_info=None,
							sequences=data)
	return result
def maximizing(data):
	result,updates=th.scan(fn=lambda i:inner_i(i),
							outputs_info=None,
							sequences=data)
	return result
def find(data):
	result,updates=th.scan(fn=lambda i:maximizing(i),
							outputs_info=None,
							sequences=data)
	return result
def find_val(data):
	data2=0.00001+data
	data=data*data2
	val0=data.shape[0]
	val1=data.shape[1]
	val2=data.shape[2]		
	return T.sum(data)/(val0*val1*val2)
def scan(b1):
	result,updates=th.scan(fn=lambda past:T.concatenate([past.reshape((-1,1)),past.reshape((-1,1))]),
									outputs_info=None,
									sequences=b1)
	return result.reshape((1,-1))
def expandimage(b1):
	result,updates=th.scan(fn=lambda past:T.concatenate([scan(past),scan(past)]),
									outputs_info=None,
									sequences=b1)
	return result.reshape((2*b1.shape[0],-1))
#y=scan(a).reshape((1,10))
def expandlayers(image):
	result,updates=th.scan(fn=lambda value:expandimage(value),
									outputs_info=None,
									sequences=image)
	return result
def check(X,i,j,a,b):
	k=T.sum(X[:,i:i+a,j:j+b])
	return k
def inner(image,ii,a,b):
	result,updates=th.scan(fn=lambda j:check(image,ii,j,a,b),
								outputs_info=None,
								sequences=T.arange(image.shape[2]-b+1))
	return result
def outer(image,a,b):
	result,updates=th.scan(fn=lambda i:inner(image,i,a,b),
								outputs_info=None,
								sequences=T.arange(image.shape[1]-a+1))
	y=T.argmax(result)
	m=y/(image.shape[1]-a+1)
	n=y%(image.shape[1]-b+1)
	z=image[:,m:m+a,n:n+b]
	return z
def change(image,val,d1,d2):
	val2=find_val(image)
	a=image.shape[1]
	b=image.shape[2]
	return ifelse(T.lt(T.cast(val,'float64'),T.cast(val2,'float64')),outer(image,d1,d2),pool_2d(image,(2,2)))
def innerbatch(layers):
	result,updates=th.scan(fn=lambda i:change(i),
							outputs_info=None,
							sequences=layers)
	return result
def magnify(batch,d1,d2):
	val=find_val(batch)
	result,updates=th.scan(fn=lambda i:change(i,val,d1,d2),
							outputs_info=None,
							sequences=batch)
	return result
def magnify2(batch,val):
	result,updates=th.scan(fn=lambda i:change(i,val),
							outputs_info=None,
							sequences=batch)
	return result
def find_threshold(images):
	result,updates=th.scan(fn=lambda i:find_val(i),
							outputs_info=None,
							sequences=images)
	return T.sum(result)/images.shape[0]
def floatX(X):
    return np.asarray(X, dtype=th.config.floatX)

def init_weights(shape):
    return th.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=th.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = th.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates
def find_val_np(data):
	data2=0.00001+data
	data=data*data2
	val0=data.shape[0]
	val1=data.shape[1]
	val2=data.shape[2]		
	return np.sum(data)/(val0*val1*val2)
def find_threshold_np(data):
	val=0.
	for i in data:
		val+=find_val_np(i)
	return val/data.shape[0]		

def model(X, w, w2,w4,w_o, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(T.cast(X,'float64'), T.cast(w,'float64')))
    #l1=magnify(l1a,13,13)
    l1 = pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(T.cast(l1,'float64'), T.cast(w2,'float64')))
    #l2=magnify(l2a,6,6)
    l2 = pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    #l3a = rectify(conv2d(T.cast(l2,'float64'), T.cast(w3,'float64')))
    #l3b = pool_2d(l3a, (2, 2))
    l3 = T.flatten(l2, outdim=2)
    #l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx
def model2(X1, w1, w12,w14,w_o1, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(T.cast(X1,'float64'), T.cast(w1,'float64')))
    #l1=magnify(l1a,9,9)
    l1 = pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(T.cast(l1,'float64'), T.cast(w12,'float64')))
    #l2=magnify(l2a,6,6)
    l2 = pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    #l3a = rectify(conv2d(T.cast(l2,'float64'), T.cast(w3,'float64')))
    #l3b = pool_2d(l3a, (2, 2))
    l3 = T.flatten(l2, outdim=2)
    #l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w14))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o1))
    return l1, l2, l3, l4, pyx    

X = T.ftensor4()
X1= T.ftensor4()
Y = T.fmatrix()
V =T.fscalar()

w = init_weights((32, 1, 3, 3))
w1 =init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w12 = init_weights((64, 32, 2, 2))
#w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((64 * 6 * 6, 625))
w14 = init_weights((64 * 4 * 4, 625))
w_o = init_weights((625, 10))
w_o1 = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x0 = model(X, w, w2,w4,w_o, 0.2, 0.5)
noise_l11, noise_l21, noise_l31, noise_l41, noise_py_x1 = model2(X1, w1, w12,w14,w_o1, 0.2, 0.5)
noise_py=noise_py_x0+noise_py_x1
l1, l2, l3, l4, py_x0 = model(X, w, w2, w4,w_o, 0., 0.)
l11, l21, l31, l41, py_x1 = model2(X1, w1, w12, w14,w_o1, 0., 0.)
py=py_x0+py_x1
y_x = T.argmax(py, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py, Y))
params = [w, w2,w4, w_o,w1,w12,w14,w_o1]
updates = RMSprop(cost, params, lr=0.01)

train = th.function(inputs=[X,Y,X1], outputs=cost, updates=updates, allow_input_downcast=True)
predict = th.function(inputs=[X,X1], outputs=y_x, allow_input_downcast=True)
print "Started training"
for i in range(10):
	print "building data trX"
	trX=np.asarray(cb(mnist.train.images),dtype=th.config.floatX)
	print "building data teX"
	teX=np.asarray(cb(mnist.test.images),dtype=th.config.floatX)
	trY=np.asarray(mnist.train.labels,dtype=th.config.floatX) 
	teY=np.asarray(mnist.test.labels,dtype=th.config.floatX)
	print trX.shape
	trx2 = np.asarray(rb(trX))
	tex2 = np.asarray(rb(teX))
	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)
	trx2 = trx2.reshape(-1,1,20,20)
	print trx2.shape
	tex2 = tex2.reshape(-1,1,20,20)
	print tex2.shape
	print i
	val1=find_threshold_np(trX)
	val2=find_threshold_np(teX)
	for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
		print start,end
		cost = train(trX[start:end], trY[start:end],trx2[start:end])
	print np.mean(np.argmax(teY, axis=1)== predict(teX,tex2))
	