---
layout: post
title: Bayesian Linear Regression
tag: Machine Learning
---

[Linear Regression](https://dlsrnsi.github.io/blog/Linear-Regression/) 에서의 추정 역시 베이지언스럽게 할 수 있다.

구하고자 하는 weight $w$의 prior가 정규분포를 따른다고 가정하자

$p(w) = N(w|m_0, S_0)$

마찬가지로, liklihood-function은 다음과 같다.

$p(t|w) = \prod^N_{n=1}N(t_n|w^T\phi(X_n), \beta^{-1})$

따라서 posteior는 다음과 같다

$p(w|t) \propto p(t|w)p(w) \propto \prod^N_{n=1}N(t_n|w^T\phi(X_n), \beta^{-1})N(w|m_0, S_0)$

w는 Exp안의 내용에만 비례하므로

$\propto \beta(t - \phi(X_n)w)^T(t - \phi(X_n)w) + S_0^{-1}(m_0^Tm_0 - 2w^Tm_0 + w^Tw)$

$= \beta t^Tt - 2w^T(\beta\phi(X_n)^T + w^T\phi(X_n)^T\phi(X_n)w) + S_0^{-1}(m_0^Tm_0 - 2w^Tm_0 + w^Tw)$

$w$에 대하여 정리하면

$w^T(S_0^{-1} + \beta\phi(X_n)^T\phi(X_n))w - 2w^T(S_0^{-1}m_0 + \beta\phi(X_n)^Tt) + const$

따라서 분산의 역(accuracy) 와 평균은 다음과 같음을 알 수 있다.

$S_N^{-1} = (S_0^{-1} + (S_0 + \beta\phi(X_n)^T\phi(X_n))$

$m_N = S_N(S_0^{-1}m_0 + \beta\phi(X_n)^Tt)$

이 때, prior가 $N(0, S_0)$를 따른다고 하면

$S_N^{-1} = S_0 + \beta\phi(X_n)^T\phi(X_n)$,   $m_N = \beta S_N\phi(X_n)^Tt$


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
x = np.arange(0,5, 0.5); y = 3 * x + 5*np.random.randn(x.shape[0]) + 2
plt.scatter(x, y)
plt.plot(x, 3*x + 2)
plt.show()
```


![png]({{ site.baseurl }}/images/Bayesian_Linear_Regression/output_2_0.png)


Linear Regression의 결과는 다음과 같다.


```python
def get_linear(x,y):
    Y = y[:,np.newaxis]
    X = np.concatenate((np.ones(len(x))[:,np.newaxis], x[:,np.newaxis]), axis=1)
    w = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(Y) ## Moore-penrose pseudo inverse matrix multiplies Y
    return w.T[0]
w = get_linear(x,y); ax = plt.subplot(1,1,1)
original = ax.plot(x, 3*x + 2, color='red',label='original');linear=ax.plot(x, w[0] + x * w[1], label='linear regression');
ax.scatter(x, y);plt.show()
```


![png]({{ site.baseurl }}/images/Bayesian_Linear_Regressionoutput_4_0.png)


linear regression(likelihood, 파란색)이 보다시피 기존함수(붉은색 선)과 차이가 꽤 나는 것을 확인할 수 있다.

Baysian Linear Regression 결과를 보도록 하자.

Prior의 $m_0 = (0,0)$으로 하고, $S_0=5$라고 한다.


```python
def get_maximum_a_posteriori(x, y, mu_0, beta, lambda_pre):
        X = np.concatenate((np.ones(len(x))[:,np.newaxis], x[:,np.newaxis]), axis=1)
        Y = y[:,np.newaxis]
        dim = X.shape[1]
        lambda_pre = np.diag(np.repeat(1./float(lambda_pre),dim))
        beta = np.diag(np.repeat(1./float(beta),dim))
        s = np.linalg.inv(lambda_pre + np.dot(beta,X.T.dot(X)))
        m_N = s.dot(lambda_pre.dot(mu_0) + beta.dot(X.T).dot(Y).T[0])
        return m_N, s
```


```python
m_N, s = get_maximum_a_posteriori(x,y,[0,0], 2, 0.2)
```


```python
for i in range(1,500):
    beta = np.random.multivariate_normal(m_N, 5*5*s)
    plt.plot(x, beta[0] + x*beta[1], alpha=0.03)
plt.plot(x, m_N[0] + x*m_N[1], color='blue'); original = plt.plot(x, 3*x + 2, color='red',label='original'); plt.scatter(x,y)
```




![png]({{ site.baseurl }}/images/Bayesian_Linear_Regressionoutput_8_1.png)


옅은 선으로 표시된 함수들은 가능한 베이지언 추정이다. 가능한 베이지언 추정$w_p$들의 평균값이 최종함수(파란색)으로 표현됨을 파악할 수 있다.

또한 likelihood function과는 달리 기존함수와 매우 비슷함을 확인할 수 있다.
