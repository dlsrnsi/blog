---
layout: post
title: Dirichlet-Multinomial Model
tag: Machine Learning
---

## Multinomial Distribution

Binomial Distribution이 동전 던지기를 계속할 때 확률 분포이면 Multinomial Distribution은 주사위 던지기를 계속하는 것과 비슷하다.

*K*-dimensional vector $$X$$를 가정하자 주사위는 6개의 값을 가지고 있으므로 $$K=6$$이며,  3이 나오는 사건은 

$$x_3 = 1$$, $$X = (0,0,1,0,0,0)^T$$ 라고 표현할 수 있다. 이 때, $$\sum^K_{k=1}x_k = 1$$이어야만 한다.

각각 사건의 확률을 $$\mu_k$$라고 할때 $$\mu = (\mu_1, \cdots , \mu_6)^T$$ 로 표현할수 있다. 이 때, $$\mu_k \geq 0 , \sum\mu_k = 1$$의 조건을 만족하여야 한다.

따라서 각각의 사건의 발생확률은 $$p(x\|\mu) = \prod^K_{k=1}\mu_k^{x_k}$$로 표현할 수 있다.

사실 베르누이 분포의 경우에는 $$K=2$$ 인 특수한 상황으로 볼 수 있다.

이제 $$N$$개의 독립적 관측데이터를 가지고 있는 데이터 셋 $$D = X_1, \cdots X_N$$를 가정하자.

이때 liklihood function은 

$$p(D\|\mu) = \prod^N_{n=1}\prod^K_{k=1}\mu_k^{x_k} = \prod^K_{k=1}\mu_k^{\sum_nx_nk} =  \prod^K_{k=1}\mu_k^{m_k}$$ 형태를 가진다.

이 떄, $$m_k = \sum_n x_{nk}$$이다. 즉 $$m_k$$는 $$k$$라는 사건이 일어난 횟수이다.

maximum liklihood를 구해보도록 하자. liklihood function에 자연로그를 취하면

$$\sum^K_{k=1}m_kln\mu_k$$가 된다. 이때 기울기 즉 도함수가 0이 되는 값은 라그랑주 승수(Lagrange multiplier)를 이용하여 구할 수 있다.

### Lagrange Multiplier

$$f(x)$$를 최적화 하고자 할때 $$g(x)=0$$인 제약조건을 이용하여 $$F(x,\lambda) = f(x) + \lambda g(x)$$를 정의하고 $$F(x)$$의 기울기가 0이 되는 값을 이용하여 찾을 수 있다.

이때 우리는 $$\sum\mu_k = 1$$이라는 제약조건을 가지고 있으므로 $$\sum\mu_k - 1=0$$으로 바꾸어 라그랑주 승수에 활용하면

$$\sum^K_{k=1}m_kln\mu_k + \lambda(\sum^K_{k=1}\mu_k - 1)$$로 바꿀 수 있다. $$\mu_k$$로 미분하였을 때 0이 될수 있도록 $$\mu_k$$를 구하면

$$\mu_k = -m_k/\lambda$$의 식을 구할 수 있다. $$\sum_k u_k=1$$임을 이용하여 $$\lambda = -N$$임을 확인할 수 있다. 따라서

$$\mu_k^{ML} = \frac {m_k} {N}$$ 이다.

### Probability Distribution

또한 probability distribution은

$$Mult(m_1, m_2, \cdots , m_k\|\mu, N) = \binom {N} {m_1m_2\cdots m_K} \prod^{K}_{k=1}\mu_k^{m_k}$$ 이며 Multinomial distribution이라고 불린다.

$$\mu = (0.3, 0.3, 0.4), N=15$$ 인 Multinormial distribution을 그려보자


```python
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import factorial
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = [0.3, 0.3, 0.4]
X = np.arange(0, 16)
Y = np.arange(0, 16)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X, dtype=np.float32)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if i+j <= 15:
            x = X[i,j]; y = Y[i,j]; w = 15 - x - y
            Z[i,j] = factorial(15)/(factorial(x)*factorial(y)*factorial(w)) * np.power(u[0], x) \
                * np.power(u[1], y) * np.power(u[2], w)
print sum(sum(Z))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.1, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
```

    0.999999984245
    


![png]({{ site.baseurl }}/images/Dirichlet_Multinomial_Model/output_1_1.png)


## Dirichlet distribution

Dirichlet distribution은 [Beta distribution](https://dlsrnsi.github.io/blog/Beta-Bernoulli-Model/)을 Multinomial distribution으로 확장시킨 것과 동일하다.

파라미터 $${\mu_k}$$의 prior는

$$p(\mu\|\alpha) \propto \prod^K_{k=1} \mu_k^{\alpha_k - 1}$$ 이 때, $$\mu_k = 1, \sum_k \mu_k= 1$$의 특성을 가진다.

이 함수가 정규화 되면,

$$Dir(\mu\|\alpha) = \frac {\Gamma(\alpha_0)} {\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)} \prod^K_{k=1}\mu_k^{\alpha_k - 1}$$ 라는 분포가 만들어진다. 이를 *Dirichlet* distribution이라고 부른다.

따라서 $$posterior$$는

$$p(\mu\|D,\alpha) \propto p(D\|\mu)p(\mu\|\alpha) \propto \prod^K_{k=1}\mu_k^{\alpha_k +m_k- 1}$$ 처럼 비례한다는 사실을 알 수 있다. 즉,

$$p(\mu\|D,\alpha) = Dir(\mu\|\alpha + m) = \frac {\Gamma(\alpha_0+N)} {\Gamma(\alpha_1+m_1)\cdots\Gamma(\alpha_K+m_K)}\prod^K_{k=1}\mu_k^{\alpha_k +m_k- 1}$$ 꼴의 분포를 가지게 된다.

마찬가지로 각각의 $$\alpha_k$$는 Pseudocount의 역할을 수행하게 된다.

