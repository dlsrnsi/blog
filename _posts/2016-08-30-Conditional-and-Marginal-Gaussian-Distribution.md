---
layout: post
title: Conditional and Marginal Distribution
tag: Machine Learning
---

## Conditional Gaussian Distribution

$$X = (X_a, X_b), X \in \mathbb{R}^N$$이고 정규분포를 따른다고 하자. 이 때, $$p(X_a \| X_b=b)$$ 역시 정규분포를 따른다.

이를 Conditional Gaussian Distribution이라고 한다.

예제를 보도록 하자. 적절한 다변수정규분포 데이터 200개를 생성한다.


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn
%matplotlib inline
```



```python
u = np.array([[0.96,-0.28],[0.28,0.96]], dtype=np.float32)
c = np.array([[2,0],[0,2]], dtype=np.float32)
c = np.dot(u,c)
data = np.random.multivariate_normal([5,5], c, 200)
plt.scatter(data[:,0],data[:,1], color='blue', linewidths=0.1, alpha=0.5)
plt.axis([0,10,0,10])
plt.show()
```


![png]({{ site.baseurl }}/images/Conditional_Marginal/output_2_0.png)


$$ 4< X_b < 6 $$ 인 $$X_a$$들을 선택하여 히스토그램을 그려보자.


```python
a = data[:,0]; b = data[:,1]
condition = (b > 4) & (b < 6)
plt.scatter(a,b, color='blue', linewidths=1, alpha=0.5)
cond_a = a[condition]; cond_b=b[condition]
plt.scatter(cond_a, cond_b, color='red', alpha=0.8)
plt.scatter(cond_a, np.zeros_like(cond_a), color='red', marker='^')
for dot1, dot2 in zip(cond_a, cond_b):
    plt.plot([dot1,dot1], [0, dot2], 'r:', linewidth=1, alpha=0.5)
plt.axis([0,10,-1,10])
q = plt.hist(cond_a, alpha=0.3, weights = 0.2 * np.ones_like(cond_a), color='green')
plt.show()
```


![png]({{ site.baseurl }}/images/Conditional_Marginal/output_4_0.png)


### 평균과 공분산

$$\sum=\begin{pmatrix} \sum_{aa}& \sum_{ab}\\ \sum_{ba} & \sum_{bb} \end{pmatrix}$$라고 하자. 이때 역행렬은$$\sum^{-1}=\Lambda=\begin{pmatrix} \Lambda_{aa}& \Lambda_{ab}\\ \Lambda_{ba} & \Lambda_{bb} \end{pmatrix}$$ 이다

따라서 정규분포의 Exp 내부를  
$$-\frac{1}{2} (x_a - u_a)^T\Lambda_{aa}(x_a - u_a)-\frac{1}{2} (x_a - u_a)^T\Lambda_{ab}(x_b - u_b)-\frac{1}{2} (x_b - u_b)^T\Lambda_{ba}(x_a - u_a)-\frac{1}{2} (x_b - u_b)^T\Lambda_{bb}(x_b - u_b)$$

로 표현할 수 있다. 이때 $$x_b=b$$는 constant 즉 상수이므로 $$x_a$$에 대한 2차식임을 확인할 수 있다.

위식의 일반적인 전개와 $$x_b=b$$에 대한 전개를 비교하면 평균과 공분산을 구할 수 있다.

$$f(x)$$의 2차항은 $$-\frac{1}{2}x_a^T\Lambda_{aa}x_a$$ 이다. 따라서 $$Cov(x_{a\|b}) = \Lambda_{aa}^{-1}$$ 이다

1차항은 $$\Lambda_{ab} = \Lambda_{ba}$$이므로 $$x_a\{\Lambda_{aa}u_a - \Lambda_{ab}(x_b - u_b)\}$$이다. 따라서 $$x_a\sum_{a\|b}^{-1}u_{a\|b} = x_a\{\Lambda_{aa}u_a - \Lambda_{ab}(x_b - u_b)\}$$ 가 된다

그러므로 $$u_{a\|b} = \Lambda_{aa}^{-1}\{\Lambda_{aa}u_a - \Lambda_{ab}(x_b - u_b)\} = u_a - \Lambda_{aa}^{-1}\Lambda_{ba}(x_b - u_b)$$

[Schur complement]: https://en.wikipedia.org/wiki/Schur_complement  "schur complement"

$$\Lambda_{ij}$$를 구하기 위하여 [Schur complement]를 활용한다.

$$\begin{pmatrix} A & B \\ C & D \end{pmatrix}^{-1} = 
\begin{pmatrix} M & -MBD^{-1} \\ -D^{-1}CM & D^{-1} + D^{-1}CMBD^{-1} \end{pmatrix}$$ 이 때,
$$ M = (A - BD^{-1}C)^{-1}$$ 이다

따라서,

$$u_{a\|b} = u_a - \sum_{ab}\sum_{bb}^{-1}(x_b - u_b)$$

$$\sum_{a\|b} = \sum_{aa} - \sum_{ab}\sum_{bb}^{-1}\sum_{ba}$$

## Marginal Gaussian Distributions

$$X = (x_a, x_b) \in \mathbb{R}^N$$ 이고 정규분포를 이룰 때, $$p(x_a) = \int p(x_a, x_b)dx_b$$이고 $$x_a, x_b$$ 각각은 정규분포를 이룬다.

이때

$$E[x_a] = u_a$$,   $$cov[x_a] = \sum_{aa}$$  이다.


```python
plt.scatter(a,b, color='blue', linewidths=1, alpha=0.5)
plt.scatter(np.zeros_like(data[:,1]),data[:,1], color='red', linewidths=1, alpha=0.2, marker='>')
for dot1, dot2 in data:
    plt.plot([0,dot1],[dot2,dot2], 'r:', linewidth=1, alpha=0.5)
plt.axis([-1,12,0,12])
plt.hist(b, orientation='horizontal', alpha=0.3, color='green', weights= 0.2 * np.ones_like(b))
plt.show()
```


![png]({{ site.baseurl }}/images/Conditional_Marginal/output_6_0.png)
