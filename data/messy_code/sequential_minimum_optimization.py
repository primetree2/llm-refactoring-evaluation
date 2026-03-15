
import os,sys,urllib.request
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs,make_circles
from sklearn.preprocessing import StandardScaler

URL="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


class SmoSVM:

    def __init__(self,train,kernel_func,alpha_list=None,cost=0.4,b=0.0,tolerance=0.001,auto_norm=True):

        self.train=train
        self.kernel=kernel_func
        self.c=np.float64(cost)
        self.b=np.float64(b)
        self.tol=np.float64(tolerance)

        if tolerance<=0.0001:
            self.tol=np.float64(0.001)

        self.auto_norm=auto_norm

        self.tags=train[:,0]

        if self.auto_norm:
            self.samples=self._norm(train[:,1:])
        else:
            self.samples=train[:,1:]

        if alpha_list is None:
            self.alphas=np.zeros(train.shape[0])
        else:
            self.alphas=alpha_list

        self.err=np.zeros(len(self.samples))
        self._eps=0.001

        self.indexes=list(range(len(self.samples)))

        self.kmat=self._build_k()

        self.unbound=[]

    def _build_k(self):

        n=len(self.samples)
        m=np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                m[i][j]=self.kernel(self.samples[i],self.samples[j])

        return m

    def _k(self,i,j):

        if isinstance(j,np.ndarray):
            return self.kernel(self.samples[i],j)

        return self.kmat[i][j]

    def _is_unbound(self,i):

        return 0<self.alphas[i]<self.c

    def _e(self,i):

        if self._is_unbound(i):
            return self.err[i]

        gx=np.dot(self.alphas*self.tags,self.kmat[:,i])+self.b
        return gx-self.tags[i]

    def _check_kkt(self,i):

        r=self._e(i)*self.tags[i]

        if (r<-self.tol and self.alphas[i]<self.c) or (r>self.tol and self.alphas[i]>0):
            return True
        return False

    def fit(self):

        changed=True

        while changed:

            changed=False

            for i1 in self.indexes:

                if not self._check_kkt(i1):
                    continue

                for i2 in self.indexes:

                    if i1==i2:
                        continue

                    y1=self.tags[i1];y2=self.tags[i2]

                    a1=self.alphas[i1];a2=self.alphas[i2]

                    e1=self._e(i1);e2=self._e(i2)

                    s=y1*y2

                    if s==-1:
                        L=max(0,a2-a1)
                        H=min(self.c,self.c+a2-a1)
                    else:
                        L=max(0,a1+a2-self.c)
                        H=min(self.c,a1+a2)

                    if L==H:
                        continue

                    k11=self._k(i1,i1)
                    k22=self._k(i2,i2)
                    k12=self._k(i1,i2)

                    eta=k11+k22-2*k12

                    if eta<=0:
                        continue

                    a2new=a2+(y2*(e1-e2))/eta

                    if a2new>H:
                        a2new=H
                    if a2new<L:
                        a2new=L

                    a1new=a1+s*(a2-a2new)

                    self.alphas[i1]=a1new
                    self.alphas[i2]=a2new

                    b1=-e1-y1*k11*(a1new-a1)-y2*k12*(a2new-a2)+self.b
                    b2=-e2-y1*k12*(a1new-a1)-y2*k22*(a2new-a2)+self.b

                    if 0<a1new<self.c:
                        self.b=b1
                    elif 0<a2new<self.c:
                        self.b=b2
                    else:
                        self.b=(b1+b2)/2

                    changed=True

    def predict(self,test,classify=True):

        if self.auto_norm:
            test=self._norm(test)

        out=[]

        for s in test:

            v=0

            for i in range(len(self.samples)):
                v+=self.alphas[i]*self.tags[i]*self._k(i,s)

            v+=self.b

            if classify:
                if v>0:
                    out.append(1)
                else:
                    out.append(-1)
            else:
                out.append(v)

        return np.array(out)

    def _norm(self,data):

        if not hasattr(self,"min"):
            self.min=np.min(data,axis=0)
            self.max=np.max(data,axis=0)

        return (data-self.min)/(self.max-self.min)

    @property
    def support(self):

        res=[]

        for i in range(len(self.alphas)):
            if self.alphas[i]>0:
                res.append(i)

        return res


class Kernel:

    def __init__(self,kernel,degree=1.0,coef0=0.0,gamma=1.0):

        self.name=kernel
        self.degree=degree
        self.coef0=coef0
        self.gamma=gamma

    def __call__(self,v1,v2):

        if self.name=="linear":
            return np.inner(v1,v2)+self.coef0

        if self.name=="poly":
            return (self.gamma*np.inner(v1,v2)+self.coef0)**self.degree

        if self.name=="rbf":
            return np.exp(-1*(self.gamma*np.linalg.norm(v1-v2)**2))

        return np.inner(v1,v2)


def test_cancer():

    if not os.path.exists("cancer.csv"):

        req=urllib.request.Request(URL,headers={"User-Agent":"Mozilla"})
        r=urllib.request.urlopen(req)
        data=r.read().decode()

        f=open("cancer.csv","w")
        f.write(data)
        f.close()

    df=pd.read_csv("cancer.csv",header=None,dtype={0:str})

    del df[df.columns[0]]

    df=df.dropna(axis=0)

    df=df.replace({"M":1.0,"B":-1.0})

    arr=np.array(df)

    train=arr[:328,:]
    test=arr[328:,:]

    test_tags=test[:,0]
    test_x=test[:,1:]

    ker=Kernel("rbf",degree=5,coef0=1,gamma=0.5)

    a=np.zeros(train.shape[0])

    model=SmoSVM(train,ker,a)

    model.fit()

    pred=model.predict(test_x)

    good=0

    for i in range(len(pred)):
        if pred[i]==test_tags[i]:
            good+=1

    print("accuracy",good/len(pred))


def demo():

    sys.stdout=open(os.devnull,"w")

    fig,axs=plt.subplots(2,2)

    _demo_linear(axs[0][0],0.1)
    _demo_linear(axs[0][1],500)
    _demo_rbf(axs[1][0],0.1)
    _demo_rbf(axs[1][1],500)

    sys.stdout=sys.__stdout__

    print("plot ready")


def _demo_linear(ax,c):

    x,y=make_blobs(n_samples=500,centers=2,n_features=2,random_state=1)

    y[y==0]=-1

    sc=StandardScaler()

    x=sc.fit_transform(x,y)

    data=np.hstack((y.reshape(500,1),x))

    k=Kernel("linear")

    m=SmoSVM(data,k,cost=c,auto_norm=False)

    m.fit()

    _plot(m,data,ax)


def _demo_rbf(ax,c):

    x,y=make_circles(n_samples=500,noise=0.1,factor=0.1,random_state=1)

    y[y==0]=-1

    sc=StandardScaler()

    x=sc.fit_transform(x,y)

    data=np.hstack((y.reshape(500,1),x))

    k=Kernel("rbf")

    m=SmoSVM(data,k,cost=c,auto_norm=False)

    m.fit()

    _plot(m,data,ax)


def _plot(model,data,ax,res=100):

    x=data[:,1];y=data[:,2];t=data[:,0]

    xr=np.linspace(x.min(),x.max(),res)
    yr=np.linspace(y.min(),y.max(),res)

    pts=np.array([(a,b) for a in xr for b in yr]).reshape(res*res,2)

    pred=model.predict(pts,False)

    grid=pred.reshape((len(xr),len(yr)))

    ax.contour(xr,yr,np.asmatrix(grid).T,levels=(-1,0,1))

    ax.scatter(x,y,c=t,cmap=plt.cm.Dark2,alpha=0.5)

    sup=model.support

    ax.scatter(x[sup],y[sup],c=t[sup],cmap=plt.cm.Dark2)


if __name__=="__main__":

    test_cancer()

    demo()

    plt.show()