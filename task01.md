```python
import torch
```

## 1.张量运算示例


```python
?torch.tensor
```


```python
#创建tensor，用dtype指定类型
a = torch.tensor(1.0,dtype = torch.float)
b = torch.tensor(1,dtype = torch.long)
c = torch.tensor(1.0,dtype = torch.int8) #可以进行强制类型的转换
print(a,b,c)
```

    tensor(1.) tensor(1) tensor(1, dtype=torch.int8)



```python
#使用制定类型的函数随机初始化指定大小的tensor
d = torch.FloatTensor(2,3)
e = torch.IntTensor([1,2,3,4])
print(d,'\n',e)
```

    tensor([[4.3799e+29, 3.0645e-41, 4.3799e+29],
            [3.0645e-41, 4.3799e+29, 3.0645e-41]]) 
     tensor([1, 2, 3, 4], dtype=torch.int32)



```python
#tensor和numpy array之间的相互转换
import numpy as np
g = np.array([[1,2,3],[4,5,6]])
h = torch.tensor(g)
print(h)
i = torch.from_numpy(g)
print(i)
j = h.numpy()
print(j)
```

    tensor([[1, 2, 3],
            [4, 5, 6]])
    tensor([[1, 2, 3],
            [4, 5, 6]])
    [[1 2 3]
     [4 5 6]]



```python
#构造Tensor的函数
k = torch.rand(2,3)
l = torch.ones(2,3)
m = torch.zeros(2,3)
n = torch.arange(0,10,2)
print(k,'\n',l,'\n',m,'\n',n)
```

    tensor([[0.2013, 0.8582, 0.6249],
            [0.1716, 0.1424, 0.2826]]) 
     tensor([[1., 1., 1.],
            [1., 1., 1.]]) 
     tensor([[0., 0., 0.],
            [0., 0., 0.]]) 
     tensor([0, 2, 4, 6, 8])



```python
#查看tensor维度
print(k.shape)
print(k.size())
```

    torch.Size([2, 3])
    torch.Size([2, 3])



```python
#tensor运算
o = torch.add(k,l)
print(o)
```

    tensor([[1.2013, 1.8582, 1.6249],
            [1.1716, 1.1424, 1.2826]])



```python
#tensor索引方式与numpy类似
print(o[:,1])
```

    tensor([1.8582, 1.1424])



```python
#改变tensor形状
print(o.view(3,2))
print(o.view(-1,2))  #-1:确定了其他维度，可以自动求出匹配的维度
```

    tensor([[1.2013, 1.8582],
            [1.6249, 1.1716],
            [1.1424, 1.2826]])
    tensor([[1.2013, 1.8582],
            [1.6249, 1.1716],
            [1.1424, 1.2826]])



```python
#tensor广播机制(注意维度)
p = torch.arange(1,3).view(1,2)
print(p)
q = torch.arange(1,4).view(3,1)
print(q)
print(p+q)
```

    tensor([[1, 2]])
    tensor([[1],
            [2],
            [3]])
    tensor([[2, 3],
            [3, 4],
            [4, 5]])



```python
#扩展&压缩tensor的维度：squeeze
print(o)
r = o.unsqueeze(1)  #在第二个维度的位置，强行加上一个维度
print(r)
print(r.size())
```

    tensor([[1.2013, 1.8582, 1.6249],
            [1.1716, 1.1424, 1.2826]])
    tensor([[[1.2013, 1.8582, 1.6249]],
    
            [[1.1716, 1.1424, 1.2826]]])
    torch.Size([2, 1, 3])



```python
s = r.squeeze(1) #注：squeeze的那一维必须是1才可以
print(s)
print(s.size())
```

    tensor([[1.2013, 1.8582, 1.6249],
            [1.1716, 1.1424, 1.2826]])
    torch.Size([2, 3])


## 2.自动求导示例
- 通过函数y=x_1 + 2*x_2 来说明pytorch自动求导的过程


```python
import torch

x1 = torch.tensor(1.0,requires_grad = True) #支持可以被求导
x2 = torch.tensor(2.0,requires_grad = True)
y = x1+2*x2
print(y)
```

    tensor(5., grad_fn=<AddBackward0>)



```python
print(x1.requires_grad)
print(x2.requires_grad)
print(y.requires_grad)
```

    True
    True
    True



```python
#查看每个导数的大小，此时应为还没有反向传播，因此导数不存在
print(x1.grad.data)
print(x2.grad.data)
print(y.grad.data)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_186/2855580583.py in <module>
          1 #查看每个导数的大小，此时应为还没有反向传播，因此导数不存在
    ----> 2 print(x1.grad.data)
          3 print(x2.grad.data)
          4 print(y.grad.data)


    AttributeError: 'NoneType' object has no attribute 'data'



```python
x1
```




    tensor(1., requires_grad=True)




```python
#反向传播后看导数大小
y = x1 + 2*x2
y.backward()
print(x1.grad.data)
print(x2.grad.data)
```

    tensor(1.)
    tensor(2.)



```python
# 导数实惠累积的，重复运行相同的命令。grad会增加
y = x1 + 2*x2
y.backward()
print(x1.grad.data)
print(x2.grad.data)
```

    tensor(2.)
    tensor(4.)


- 所以每次计算前需要清除当前的导数值避免积累，这一功能可以通过optimizer实现


```python
#尝试，如果不允许求导，会出现什么情况
x1 = torch.tensor(1.0,requires_grad = False) 
x2 = torch.tensor(2.0,requires_grad = False)
y = x1+2*x2
y.backward() #出现报错
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /tmp/ipykernel_186/2963658061.py in <module>
          3 x2 = torch.tensor(2.0,requires_grad = False)
          4 y = x1+2*x2
    ----> 5 y.backward()
    

    ~/miniconda3/envs/openvivo_env_py37/lib/python3.7/site-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
        394                 create_graph=create_graph,
        395                 inputs=inputs)
    --> 396         torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
        397 
        398     def register_hook(self, hook):


    ~/miniconda3/envs/openvivo_env_py37/lib/python3.7/site-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        173     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        174         tensors, grad_tensors_, retain_graph, create_graph, inputs,
    --> 175         allow_unreachable=True, accumulate_grad=True)  # Calls into the C++ engine to run the backward pass
        176 
        177 def grad(


    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn



```python
y #可以前向传播，不能反向传播
```




    tensor(5.)


