
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import animation
import pickle

#%% 1.Loading Data and preprocessing ############################
#################################################################
datos = pd.read_table("allCaracTraining.dat", sep=' ') #File with Training Data
X, y = datos[datos.keys()[1:-1]], datos[datos.keys()[-1]]

# Selecting features by deliting not important ones
X= X.drop(['r_value_min', 'slope_min', 'eta', 'DeltaM', 'med', 'LSlope', 'reDSign', 'rbLeonsign', 'per_3', 'tm'], axis=1)

# Selecting Classes by deliting not important ones
fy_ind=y[(y == 'TAL') | (y == 'PAC') |(y == 'PAD') |(y == 'PC') |(y == 'PD') |(y == 'ME') |(y == 'T1') | (y == 'NoC')].index

X.drop(fy_ind, inplace = True) 
y.drop(fy_ind, inplace = True) 

y= y.to_numpy()

#scaling data
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

X_Cte=X.copy()
dim=len(X.T) #dimension of X
indices=np.loadtxt('FI_ind_18.txt') #Indices of features ordered in by highest Feature Importance.
indices=indices.astype(int)
indicesCte=indices.copy()

num_classes= len(set(y)) #number of classes

#%% 2.Functions ####################################################
####################################################################
def dG(G,x1,x2): #Distance between point x1 and x2 under matrix G
    mG=np.reshape(G, (dim,dim))
    mGT=mG.T
    mA=np.matmul(mG,mGT)
    return np.matmul(np.matmul((x2-x1), mA),np.transpose(x2-x1))

def g(A): #Function to minimize
    sS=0
    sD=0
    for p in S:
        sS+=dG(A,Xn[p[0]],Xn[p[1]])
    for p in D:
        sD+=dG(A,Xn[p[0]],Xn[p[1]])
    return sS-np.log(np.sqrt(sD))


def ClustAcc(tr,pr): # Clustering Accuracy A_c
    s=0
    for i in range(len(tr)):
        for j in range(i+1,len(tr)):
            if tr[i]==tr[j]:
                if pr[i]==pr[j]:
                    s+=1
                    continue
            if tr[i]!=tr[j]:
                if pr[i]!=pr[j]:
                    s+=1
    m=len(tr)
    den=0.5*m*(m-1)
    return s/den

Nfeval = 1
def callbackF(Xi): # For optimization
    global Nfeval
    X_met=np.matmul(Xn,np.reshape(Xi, (dim,dim)))
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(X)
    kmlabels=kmeans.labels_
    kmeansMet = KMeans(n_clusters=num_classes, random_state=0).fit(X_met)       
    kmlabelsMet=kmeansMet.labels_
    print ('{0:4d}  {1: 3.6f}  {2: 3.6f}  {3: 3.6f}'.format(Nfeval, g(Xi), ClustAcc(y, kmlabels), ClustAcc(y, kmlabelsMet)))
    Nfeval += 1    
    
#%% 3.Metric Optimization ####################################################
##############################################################################
for i in range(len(indicesCte)-1): #Creating sets S and D (similar and different)
    indices=indicesCte.copy()
    Xn=X.copy()
    Xn=Xn[:,indices]
    #Similar and Different Sets
    S=[]
    D=[]
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            pair=[i,j]
            r=np.random.rand()
            if r<0.1:
                if y[i] == y[j]:
                    S.append(pair)
                else:
                    D.append(pair)   
    dim=len(Xn.T)
    Iden=np.zeros(dim**2)
    
    #Optimization (Long processing time)
    for i in range(dim):
            Iden[i*(dim+1)]=1
    print  ('{0:4s}   {1:9s}  {2:9s}  {3:9s}'.format('Iter', 'f(X)','K-means Org', 'K-means Met'))
    minm = minimize(g, Iden,options={'maxiter': 50}, callback=callbackF)
    res=minm.x  
    f = open("metricas.txt", "a") #Metrics are saved on this file
    f.write(' \n'+ str(res) +'\n' )
    f.close()
    indices=indices[:-1]


#%% 4.Applying Metrics ####################################################
###########################################################################

with open('metricas.txt') as f:
    str_arr = ' '.join([l.strip() for l in f])

mets = np.asarray(str_arr.split(' '), dtype=str)

mets=mets[mets != '']

matn=18
matrices={}
for i in range(len(mets)):
    if mets[i][0]=='[':
        mets[i]=mets[i][1:]
        matrices["G{0}".format(matn)]=[]
    if mets[i][-1]==']':
        mets[i]=mets[i][:-1]
        matn=matn-1
        matrices["G{0}".format(matn+1)].append(float(mets[i]))
        continue
    matrices["G{0}".format(matn)].append(float(mets[i]))
    
matn=18
X_mets={}
indices=indicesCte.copy()
for i in matrices:
    Xn=X.copy()
    Xn=Xn[:,indices]
    d= int(np.sqrt(len(matrices[i])))
    matrices[i]=np.reshape(matrices[i], (d,d))
    X_mets["X_met{0}".format(matn)]=np.matmul(Xn,matrices[i])   
    matn=matn-1
    indices=indices[:-1]

#%% 5.Clustering Acuracies calculation ###########################
##################################################################
acc=[]
for i in X_mets:
    X_met=X_mets[i]
    kmeansMet = KMeans(n_clusters=num_classes, random_state=0).fit(X_met)
    kmlabelsMet=kmeansMet.labels_
    acc.append(ClustAcc(y, kmlabelsMet))

#%% 6.Plots #####################################################
#################################################################
#This section is used to obtain the animated plots of the data points' first 3 dimensions 
y_ints = dict([(j,i+1) for i,j in enumerate(sorted(set(y)))])
y_ints= [y_ints[i] for i in y]
kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(X)
kmlabels=kmeans.labels_

fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
def init():
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_ints,cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(str(ClustAcc(y, kmlabels))+'  18 features')
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate, creating mp4 files for the animations of the data.
print('animating original')
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
#original data
print('saving')
anim.save('TESSNoM.mp4', fps=30)
print('done')

# with metric and lowering dimension
d=18
for i in X_mets: 
    #Metric
    if i == 'X_met2':
        break
    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)
    def init():
        ax.scatter(X_mets[i][:, 0], X_mets[i][:, 1], X_mets[i][:, 2], c=y_ints,cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(str(acc[18-d])+'  '+str(d)+' features')
        return fig,
    
    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,
    print('animating '+str(d))
    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    print('saving')
    # Save
    anim.save('Met'+str(d)+'.mp4', fps=30)
    print('done')
    d=d-1

    
#%% 7.Classification ######################################################
###########################################################################

# Methods: DT, KNN, y RF
# the following are hyper parameters to be optimized 
cp1=np.linspace(start=1, stop=499, num=499)*1E-4
cp2=np.linspace(start=5, stop=10, num=6)*0.01
cp3=np.linspace(start=2, stop=6, num=5)*0.1
cp_candidates= np.concatenate((cp1,cp2,cp3), axis = None)

n_candidates= np.linspace(start=3, stop=178, num=176).astype(int)

digits = np.arange(start=1, stop=len(indices))

DT = DecisionTreeClassifier(ccp_alpha=1e-4, max_depth=30, random_state=42)
KNN = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=500, random_state=42)
for metodo in ['DT', 'KNN', 'RF']:
    if metodo == 'DT':
        tuned_parameters = [{'ccp_alpha' : cp_candidates}]
        method = DT
    if metodo == 'KNN':    
        tuned_parameters = [{'n_neighbors' : n_candidates}]
        method = KNN
    if metodo == 'RF':
        tuned_parameters = [{'max_features' : digits}]
        method = rfc
    scor = 'accuracy'
    scorer = accuracy_score
    met=[]
    # this following loop is for lowering dimension according to FI
    indices=indicesCte.copy()
    for i in range(len(indicesCte)-1):
        Xn=X.copy()
        Xn=Xn[:,indices]
        grid = GridSearchCV(method, tuned_parameters, scoring=scor, n_jobs=6, cv=10, verbose=1,)
        grid.fit(Xn, y)
        print(metodo+'_'+'acc'+str(len(indicesCte)-i))
        y_pred = cross_val_predict(grid.best_estimator_, Xn, y, cv=10, verbose=1)
        with open(metodo+'_'+'acc'+str(len(indicesCte)-i)+'I.pkl', 'wb') as fid:
            pickle.dump(grid.best_estimator_, fid) 
        met.append(scorer(y, y_pred))
        f = open("Informe.txt", "a") #Summary is saved of this file
        f.write(' \n'+metodo+'_'+'acc'+str(len(indicesCte)-i)+'    '+str(grid.best_params_)+'    '+str(accuracy_score(y, y_pred))+ '    '+ str(cohen_kappa_score(y, y_pred))+'\n' )
        f.close()
        recalls=recall_score(y, y_pred, average=None)
        for i in range(len(recalls)):
            f = open("Informe.txt", "a")
            f.write(str(recalls[i])+'\n')
            f.close()
        indices=indices[:-1]
    
    #same process with metrics
    metM=[]
    i=0
    for j in X_mets:
        Xn=X_mets[j]
        grid = GridSearchCV(method, tuned_parameters, scoring=scor, n_jobs=6, cv=10, verbose=1,)
        grid.fit(Xn, y)
        print(metodo+'Met_'+'acc'+str(len(indicesCte)-i))
        y_pred = cross_val_predict(grid.best_estimator_, Xn, y, cv=10, verbose=1)
        with open(metodo+'Met_'+'acc'+str(len(indicesCte)-i)+'I.pkl', 'wb') as fid:
            pickle.dump(grid.best_estimator_, fid) 
        metM.append(scorer(y, y_pred))
        f = open("InformeMet.txt", "a")
        f.write(' \n'+metodo+'Met_'+'acc'+str(len(indicesCte)-i)+'    '+str(grid.best_params_)+'    '+str(accuracy_score(y, y_pred))+ '    '+ str(cohen_kappa_score(y, y_pred))+'\n' )
        f.close()
        recalls=recall_score(y, y_pred, average=None)
        for k in range(len(recalls)):
            f = open("InformeMet.txt", "a")
            f.write(str(recalls[k])+'\n')
            f.close()
        i=i+1
    plt.figure()    
    plt.plot(np.linspace(18,2,num=17),met,label='Without Metric')
    plt.plot(np.linspace(18,2,num=17),metM, label='With Metric')
    plt.legend()
    plt.xlim((20,0))
    plt.xlabel('Features')
    plt.ylabel('acc')
    plt.title(metodo)
    plt.savefig(metodo+'_'+'acc')