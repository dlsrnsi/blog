---
layout: post
title: Fisher's Linear Discriminant
tag: Machine Learning
---

정규분포를 따르는 데이터 집합 $$A,B$$가 존재한다고 하자.

각각의 평균은 $$(5,10),(10,5)$$ 이고, 분산은 $$\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$ 라고 하자.

전체 데이터 집합에서 A와 B를 분류해보고자 한다.


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
var = [[2,0],[0,2]]
a = np.random.multivariate_normal([5,10], var, 30).T
b = np.random.multivariate_normal([7,7], var, 30).T
```


```python
plt.scatter(a[0,:], a[1,:], marker="^", color='red')
plt.scatter(b[0,:], b[1,:], color='blue')
plt.axis((0,15,0,15))
plt.hist(a[0,:],color='red', alpha=0.3)
plt.hist(b[0,:], color='blue', alpha=0.3)
plt.show()
```


![png]({{ site.baseurl }}/images/Fisher_Linear_Discriminant/output_3_0.png)


히스토그램을 그려보면 히스토그램 Bar가 겹치는 것을 볼 수 있다. 즉 이 데이터는 하나의 변수에 대해서 Marginal하게 분리할 수 없다.

$$rotation matrix = \begin{bmatrix} cos(\theta) & -sin(\theta) \\ sin(\theta) & cos(\theta) \end{bmatrix}$$, 즉 회전행렬을 이용하여

45도를 회전한뒤 히스토그램을 그려보도록 하자


```python
rotate_matrix = lambda x: np.array([[np.cos(x), -np.sin(x)],[np.sin(x), np.cos(x)]], dtype=np.float32)
```


```python
r_45 = rotate_matrix(np.pi/4)
r_45
```




    array([[ 0.70710677, -0.70710677],
           [ 0.70710677,  0.70710677]], dtype=float32)




```python
a_45 = r_45.dot(a)
b_45 = r_45.dot(b)
plt.scatter(a_45[0,:], a_45[1,:], marker="^", color='red')
plt.scatter(b_45[0,:], b_45[1,:], color='blue')
plt.hist(a_45[0,:],color='red', alpha=0.3)
plt.hist(b_45[0,:], color='blue', alpha=0.3)
plt.show()
```


![png]({{ site.baseurl }}/images/Fisher_Linear_Discriminant/output_7_0.png)


히스토그램이 분리된것을 볼 수 있다.

그러나 모든 데이터가 다짜고짜 45도를 돌린다고 해서 잘 분리되는 것은 아니다.

Fisher의 Linear Discriminant는 이러한 방법의 해결책을 알려준다.

선형 판별식의 가장 간단한 형태는 다음과 같다

$$y(x) = w^Tx+w_0$$

일반적으로 2-class 문제에서는 if $$y(x)\geq0=1$$ else $$0$$로 클래스를 분류한다

따라서 이 때, 결정 경계면 (boundary)은 $$y(x)=0$$이다

점 $$x_A,x_B$$가 결정 경계면 위에 있다고 가정하자.

이 때, $$y(x_A)=y(x_B)=0$$이다. 따라서 $$w^T(x_A-x_B)=0$$이고 여기서 $$w$$는 결정 경계선에 수직(orthogonal)함을 알 수 있다.

두 데이터의 평균값 $$m_A,m_b$$를 가정하자


```python
a_m = a.mean(axis=1)
b_m = b.mean(axis=1)
```

결정 경계선에 사영(projection)된 각각의 그룹은 분산 $$s_A^2, s_B^2$$을 가지고 있게된다.

$$s^2_A = \sum_{n\in C_A}(w^Tx-w^Tm_A)^2$$,
$$s^2_B = \sum_{n\in C_B}(w^Tx-w^Tm_B)^2$$


```python
a_v = a.var(axis=1)
b_v = b.var(axis=1)
```

이때 사영된 그룹간의 평균의 차가 크면 클수록 그룹내의 분산이 작아지게 된다. 자세한 내용은 ANOVA에서 다룰 것이다.

따라서 그룹간 차를 줄이고 그룹내 분산이 작아지는 기준을 만든다

$$J(w) = \frac {(w^Tm_A-w^Tm_B)^2} {s_A^2+s_B^2}$$

이를 다르게 표현하면,

$$J(w) = \frac {w^TS_Bw} {w^TS_Ww}$$,   where  
$$S_B = (m_A-m_B)(m_A-m_B)^T, S_W = \sum_{n \in C_A}(x_n - m_A)(x_n -m_A)+\sum_{n \in C_B}(x_n - m_B)(x_n -m_B)$$

이를 w에 대해서 미분하면

$$(w^TS_Bw)S_Ww=(w^TS_Ww)S_Bw$$ 가 나오게 되고, w에 대해서 정리하면, $$(w^TS_Bw),(w^TS_Ww)$$는 스칼라이므로

$$w=rS_W^{-1}(m_A-m_B)= \frac {m_A-m_B} {s_A^2+s_B^2}$$ 가 나오게 된다


```python
w = (a_m - b_m)/(a_v + b_v)
new_w = w.dot(rotate_matrix(np.pi/2))
plt.figure(figsize=(5,5))
x = np.linspace(0,15)
y = (w[1]/w[0])*x + (a_m[1]+b_m[1])/2 - (w[1]/w[0])*(a_m[0]+b_m[0]) /2
y2 = (new_w[1]/new_w[0])*x + (a_m[1]+b_m[1])/2 - (new_w[1]/new_w[0])*(a_m[0]+b_m[0]) /2
plt.scatter(a[0,:], a[1,:], marker="^", color='red')
plt.scatter(b[0,:], b[1,:], color='blue')
plt.plot(x,y, color='green', )
plt.plot(x,y2, color='grey')
plt.axis((0,15,0,15))
plt.plot()
```




    []




![png]({{ site.baseurl }}/images/Fisher_Linear_Discriminant/output_13_1.png)


이때 결정경계선은 붉은색 선으로 그려졌고, w는 초록색으로 그려졌다.

위에서 언급했듯이 결정경계선과 w는 직교한다.


```python
print a_m, b_m
```

    [ 5.07713923  9.91909568] [ 7.12310527  6.6260401 ]
    


```python
angle = np.arctan((new_w[1]/new_w[0]))
```


```python
angle
```




    0.64476264981417719




```python
dis_m = rotate_matrix(angle)
a_dis = dis_m.dot(a)
b_dis = dis_m.dot(b)
plt.scatter(a_dis[0,:], a_dis[1,:], marker="^", color='red')
plt.scatter(b_dis[0,:], b_dis[1,:], color='blue')
plt.hist(a_dis[0,:],color='red', alpha=0.3)
plt.hist(b_dis[0,:], color='blue', alpha=0.3)
plt.show()
```


![png]({{ site.baseurl }}/images/Fisher_Linear_Discriminant/output_18_0.png)


결정 경계선만큼 회전하여 히스토그램을 그려보면 데이터가 분리되었음을 볼 수 있다.

