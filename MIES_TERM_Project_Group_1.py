# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 19:09:52 2021

@author: ravij
"""
#------------------------------------------------------------------------------
import math as mt
import os
import string
from datetime import datetime
import numpy as np
import itertools
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#for week 1
#Authenticate User :32005
#lines = [line.rstrip('\t\n') for line in open('MIES/17EC32005week1.txt')]
#f_list = [event.split('\t') for event in lines]
#lines1 = [line.rstrip('\t\n') for line in open('MIES/17EC34003week1.txt')]
#
#line2 = [line.rstrip('\t\n') for line in open('MIES/17EC35023week1.txt')]
#lines1 = lines1 + line2
#f_list1 = [event.split('\t') for event in lines1]

#------------------------------------------------------------------------------

#Authenticate User :34003
#for Week 1+2
#lines = [line.rstrip('\t\n') for line in open('MIES/Mouselog_17EC34003.txt')]
#f_list = [event.split('\t') for event in lines]
#lines1 = [line.rstrip('\t\n') for line in open('MIES/Mouselog_17EC32005.txt')]
#
#
#line2 = [line.rstrip('\t\n') for line in open('MIES/Mouselog_17EC35023.txt')]
#lines1 = lines1 + line2
#f_list1 = [event.split('\t') for event in lines1]

#------------------------------------------------------------------------------
#for week 1+2+3
#Authenticate User :32005

#lines = [line.rstrip('\t\n') for line in open('MIES/17EC32005week123.txt')]
#f_list = [event.split('\t') for event in lines]
#lines1 = [line.rstrip('\t\n') for line in open('MIES/17EC34003week123.txt')]
#
#line2 = [line.rstrip('\t\n') for line in open('MIES/17EC35023week123.txt')]
#lines1 = lines1 + line2
#f_list1 = [event.split('\t') for event in lines1]

#------------------------------------------------------------------------------
#for week 1+2+3+4
#Authenticate User :34003

lines = [line.rstrip('\t\n') for line in open('MIES/17EC34003week1234.txt')]
f_list = [event.split('\t') for event in lines]
lines1 = [line.rstrip('\t\n') for line in open('MIES/17EC32005week1234.txt')]

line2 = [line.rstrip('\t\n') for line in open('MIES/17EC35023week1234.txt')]
lines1 = lines1 + line2
f_list1 = [event.split('\t') for event in lines1]

#%%

#time Difference Function:

def timediff(t1,t2):
#    t1=(t1 + '000')
#    t2=(t2 + '000')
    day1=datetime.strptime(t1, "%d:%m:%Y:%H:%M:%S:%f")
    day2=datetime.strptime(t2, "%d:%m:%Y:%H:%M:%S:%f")
    sec = (day2-day1).total_seconds()
    return(sec)

#for authenticate User 
#Dmm  = Points Having Mouse Movements	
dMM = [x for x in f_list if 'MM' in x]
#DMP = Points having Mouse Point
dMP = [x for x in f_list if 'MP' in x]
#Dmr = Points having Mouse Release
dMR = [x for x in f_list if 'MR' in x]
#DMwM = Points Having Mouse Wheel Movement
dMWM = [x for x in f_list if 'MWM' in x]

#for User 2
#Dmm  = Points Having Mouse Movements	
dMM1 = [x for x in f_list1 if 'MM' in x]
#DMP = Points having Mouse Point
dMP1 = [x for x in f_list1 if 'MP' in x]
#Dmr = Points having Mouse Release
dMR1 = [x for x in f_list1 if 'MR' in x]
#DMwM = Points Having Mouse Wheel Movement
dMWM1 = [x for x in f_list1 if 'MWM' in x]


#%%

#scr  = scroll points 
scr = [item[-2] for item in dMWM]
scr1 = [item1[-2] for item1 in dMWM1]

#Xmmr = X axis moveent points
xmmr = np.array([int(item[1]) for item in dMM])
xmm1 = [int(item[1]) for item in dMM1]
#ymmr = y axis moveent points
ymmr = np.array([int(item[2]) for item in dMM])
ymm1 = [int(item[2]) for item in dMM1]
#tmmr = time points for Mouse movement
tmmr = [item[-1] for item in dMM]
tmm1 = [item[-1] for item in dMM1]
#tmwmr = time for Mouse wheel movements
tmwmr = [item1[-1] for item1 in dMWM]
tmwm1 = [item1[-1] for item1 in dMWM1]


#------------------------------------------------------------------------------
#angular Motion velocity Proportion function:

def angular_motion(scr,tmw):
#    timdif=[]
    ang_val=[]
    scr = np.asarray(scr,dtype = np.float64)
#    timdif.append(timediff(tmw[i],tmw[i+1]))
    for i in range(len(scr)-1):
        if timediff(tmw[i],tmw[i+1])!=0:
            ang_val.append(scr[i]/timediff(tmw[i],tmw[i+1]));
        else:
            ang_val.append(0);
    return ang_val
        
    
#calling angular Motion velocity Proportion function:

ang= angular_motion(scr,tmwmr)
ang1= angular_motion(scr1,tmwm1)

#------------------------------------------------------------------------------

#Function to find the Euclidean distance Travelled and timedifference 
def fet_val(xmmr,tmmr,ymmr):
    d =[]
    timdif=[]
    for i in range(len(xmmr)-1):
        timdif.append(timediff(tmmr[i],tmmr[i+1]))
        d.append(mt.sqrt((xmmr[i+1]-xmmr[i])**2+(ymmr[i+1]-ymmr[i])**2))
#        d.append(abs(xmmr[i+1]-xmmr[i])+abs(ymmr[i+1]-ymmr[i]))
#        if timediff(tmmr[i],tmmr[i+1])!=0:    
#            vmmx.append((xmmr[i+1]-xmmr[i])/timediff(tmmr[i],tmmr[i+1]))
#        else:
#            vmmx.append(0)
    return d,timdif

#xmmr=np.array(xmmr)

#nby = mt.floor(len(ymmr)/bs)
#nbt = mt.floor(len(tmmr)/bs)
    
#Calling Function
d,timdif= fet_val(xmmr,tmmr,ymmr)
d1,timdif1 = fet_val(xmm1,tmm1,ymm1)

#Batch Size Set to 1000
bs = 1000
#No of Batches Formed
nbd = mt.floor(min(len(d1),len(d))/bs)

#xmm = np.zeros()

#Function For Batch Formation:
#------------------------------------------------------------------------------

def batch_form(nb,bs,fet):
    fet = fet[0:(nb*bs)]
    bt=np.reshape(fet,(nb,bs))
    return bt

#Old Function For Batch Formation:
    
#def batch_form(nb,bs,p,fet):
#    bt=np.zeros((nb,bs,p))
#    for j in range(bs):
#        for i in range(nb):
#            p1=mt.floor(i*bs)
#            p2=mt.floor((i+1)*bs)
#            bt[i][0:(i+1)*bs][:]=np.reshape(fet[p1:p2],(bs,1))
#    return bt


#Function Calling :
db = batch_form(nbd,bs,d)
tb = batch_form(nbd,bs,timdif)
db1 = batch_form(nbd,bs,d1)
tb1 = batch_form(nbd,bs,timdif1)

#Batch Size For Mouse Wheel Movement:
bs1 = mt.floor(min(len(dMWM),len(dMWM1))/nbd);

#No. of batch formed 
nba = mt.floor(min(len(ang),len(ang1))/bs1)

#batch Formation for Angular velocity:
an = batch_form(nba,bs1,ang)
an1 = batch_form(nba,bs1,ang1)
#------------------------------------------------------------------------------
#Function to find Mean and Standard Deviation:
def mean_std(X,Y,N):
    mX = [];
    mY =[];
    sX =[];
    sY = [];
    for i in range (nbd):
        mX.append(np.mean(X[i,:]))
        mY.append(np.mean(Y[i,:]))
        sX.append(np.std(X[i,:]))
        sY.append(np.std(Y[i,:]))
    return mX,mY,sX,sY

#Function Calling :

mdb,mtb,sdb,stb = mean_std(db,tb,nbd)
mdb1,mtb1,sdb1,stb1 = mean_std(db1,tb1,nbd)
man,man1,stan,stan1 = mean_std(an,an1,nbd)

#------------------------------------------------------------------------------
##Following Process For Finding Velocity And Acceleration As Feature:

#user 1
#tim =[]
#vmmx=[]
#dx =[]
#dy =[]
#t=0;
#for i in range(len(xmmr)-1):
#    t = t + timediff(tmmr[i],tmmr[i+1])
#    tim.append(t)
#    dx.append(xmmr[i+1]-xmmr[i])
#    if timediff(tmmr[i],tmmr[i+1])!=0:    
#        vmmx.append((xmmr[i+1]-xmmr[i])/timediff(tmmr[i],tmmr[i+1]))
#    else:
#        vmmx.append(0)
#ammx=[]
#for i in range(len(vmmx)-1):
#    if timediff(tmmr[i],tmmr[i+1])!=0:    
#        ammx.append((vmmx[i+1]-vmmx[i])/timediff(tmmr[i],tmmr[i+1]))
#    else:
#        ammx.append(0)
#vmmy=[]
#for i in range(len(ymmr)-1):
#    dy.append(ymmr[i+1]-ymmr[i])
#    if timediff(tmmr[i],tmmr[i+1])!=0:    
#        vmmy.append((ymmr[i+1]-ymmr[i])/timediff(tmmr[i],tmmr[i+1]))
#    else:
#        vmmy.append(0)
#        
#ammy=[]
#for i in range(len(vmmy)-1):
#    if timediff(tmmr[i],tmmr[i+1])!=0:    
#        ammy.append((vmmy[i+1]-vmmy[i])/timediff(tmmr[i],tmmr[i+1]))
#    else:
#        ammy.append(0)

#------------------------------------------------------------------------------
##user 2
#tim1 =[]
#vmm1x=[]
#dx1 =[]
#dy1=[]
#t=0;
#for i in range(len(xmm1)-1):
#    t = t + timediff(tmm1[i],tmm1[i+1])
#    tim1.append(t)
#    dx1.append(xmm1[i+1]-xmm1[i])
#    if timediff(tmm1[i],tmm1[i+1])!=0:    
#        vmm1x.append((xmm1[i+1]-xmm1[i])/timediff(tmm1[i],tmm1[i+1]))
#    else:
#        vmm1x.append(0)
#amm1x=[]
#for i in range(len(vmm1x)-1):
#    if timediff(tmm1[i],tmm1[i+1])!=0:    
#        amm1x.append((vmm1x[i+1]-vmm1x[i])/timediff(tmm1[i],tmm1[i+1]))
#    else:
#        amm1x.append(0)
#vmm1y=[]
#for i in range(len(ymm1)-1):
#    dy1.append(ymm1[i+1]-ymm1[i])
#    if timediff(tmm1[i],tmm1[i+1])!=0:    
#        vmm1y.append((ymm1[i+1]-ymm1[i])/timediff(tmm1[i],tmm1[i+1]))
#    else:
#        vmm1y.append(0)        
#        
#amm1y=[]
#for i in range(len(vmm1y)-1):
#    if timediff(tmm1[i],tmm1[i+1])!=0:    
#        amm1y.append((vmm1y[i+1]-vmm1y[i])/timediff(tmm1[i],tmm1[i+1]))
#    else:
#        amm1y.append(0)
    
    
#------------------------------------------------------------------------------
##IF Normaliation Is Required :
        
#xmm =np.array(xmm)
#xmm1 =np.array(xmm1)
#vmm =np.array(vmm)
#vmm1 =np.array(vmm1)
#all Means 

##Normalization ;
#for i in range(len(xmm)):
#    mean=np.mean(xmm);
#    SD = np.sqrt(np.sum((xmm[i]-mean)**2)/len(xmm));
#    xmm[i]=(xmm[i]-mean)/SD;
#for i in range(len(xmm1)):
#    mean=np.mean(xmm1);
#    SD = np.sqrt(np.sum((xmm1[i]-mean)**2)/len(xmm1));
#    xmm1[i]=(xmm1[i]-mean)/SD;
#    
#for i in range(len(vmm)):
#    mean=np.mean(vmm);
#    SD = np.sqrt(np.sum((vmm[i]-mean)**2)/len(xmm));
#    vmm[i]=(vmm[i]-mean)/SD;
#for i in range(len(vmm1)):
#    mean=np.mean(vmm1);
#    SD = np.sqrt(np.sum((vmm1[i]-mean)**2)/len(vmm1));
#    vmm1[i]=(vmm1[i]-mean)/SD;
    
#for i in range(len(xmm)-1):
#    train.append([xmm[i],vmm[i]])
#for i in range(len(xmm1)-1):
#    train.append([xmm1[i],vmm1[i]])

#------------------------------------------------------------------------------
##Resultant Velocity and acceleration Finding:
#st = min(len(vmmx),len(vmm1x),len(ammx),len(amm1x))
#vm =[]
#am=[]
#vm1=[]
#am1=[]    
#for i in range(st):
#    vm.append([mt.sqrt(vmmx[i]**2+vmmy[i]**2)])
#for i in range(st):
#    vm1.append([mt.sqrt(vmm1x[i]**2+vmm1y[i]**2)])
#
#for i in range(st):
#    am.append([mt.sqrt(ammx[i]**2+ammy[i]**2)])
#for i in range(st):
#    am1.append([mt.sqrt(amm1x[i]**2+amm1y[i]**2)])

##batch Size:
#bs =1000;
##No of batches
#nbv= mt.floor(st/bs);
#
#Batch Formation for velocity and acceleration:
#nvb = batch_form(nbv,bs,p,vm)
#nvb1 = batch_form(nbv,bs,p,vm1)
#mvb,mvb1,svb,svb1 = mean_std(nvb,nvb1,nbv)
#for i in range(st):
#    train.append([ammx[i],ammy[i]])
#for i in range(st):
#    train.append([amm1x[i],amm1y[i]])
#   
#minlen = min(len(mdb),len(man))
#%%

train =[]
addi = []
trlen = mt.floor(len(mdb)*0.8)  
tslen = len(mdb)-trlen 

#Splitting 80% For Training :
def train_split(trlen):
    for i in range(trlen):
#        train.append([mtb[i],mdb[i],stb[i],sdb[i],man[i],stan[i]])
#        train.append([mdb[i],sdb[i],man[i],stan[i]])
#        train.append([sdb[i],stan[i]])
#        addi.append(sdb[i],stan[i])
    #    print(addi[i])
        train.append([mdb[i],man[i]])
        
    for i in range(trlen):
#        train.append([mtb1[i],mdb1[i],stb1[i],sdb1[i],man1[i],stan[i]])
#        train.append([mdb1[i],sdb1[i],man1[i],stan1[i]])
#        train.append([sdb1[i],stan1[i]])
#        add1.append(sdb1[i],stan1[i])
    #    print(add1[i])
        train.append([mdb1[i],man1[i]])
        
    tr =np.array(train);
    return tr
tr = train_split(trlen)
#%%

#Plotting For Different Features: 
s= 2
e = 3
plt.figure(0)    
#plt.scatter(tr[1:st,0],tr[1:st,1],s=20,c='red')
plt.scatter(tr[1:trlen,s],tr[1:trlen,e],s=1,c='red')
#plt.figure(1)
plt.scatter(tr[trlen:2*trlen,s],tr[trlen:2*trlen,e],s =1,c='green')

#%%

#Training Using SVM :
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#For Using Linear Kernel
#clf = SVC(C = 1e5, kernel = 'linear')

#For Radial Basis Function As Kernel
clf = SVC(C = 1e5, kernel = 'rbf')

#Train Set :
X = np.array(train)

#Labels:
y1=np.zeros(trlen)
y2 = np.ones(trlen)
y = np.concatenate([y2,y1])

#Fitting The Train set and Labels :
clf.fit(X, y) 

#print('w = ',clf.coef_)
#print('b = ',clf.intercept_)
#w = clf.coef_
#b = clf.intercept_
#
#print(w.shape)
#print(b.shape)

#%%


#Five Fold Validation:
    
f1m = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
x = np.linspace(1,5,len(acc))
plt.bar(x,acc*100)
plt.title('Accuracy for Five folds')
plt.xlabel('Folds')
plt.ylabel('Accuracy')
#Printing ACCURACY and F1 Score:
ACCU = np.mean(acc)
F1 = np.mean(f1m)
print('Accuracy :',ACCU*100,'%')
print('F1 Score:',F1)

#------------------------------------------------------------------------------    
#%%

#For Using 20% data as Test set :

test=[];
timtest=[]
t=0;
stp = trlen+1
etp = len(mdb)

for i in range(stp,etp):
#    t = t + timediff(tmm[i],tmm[i+1])
#    timtest.append(t)
#    test.append([timtest[i-3001],vmmy[i]])
#    test.append([mtb[i],mdb[i],stb[i],sdb[i],man[i],stan[i]])
#    addi.append(stan[i]+sdb[i])
#    print(addi[i])
#    test.append([mdb[i],man[i],mvb[i],svb[i]])
    test.append([mdb[i],man[i]])
#    test.append([mdb[i],sdb[i],man[i],stan[i]])
for i in range(stp,etp):
#    test.append([mtb1[i],mdb1[i],stb1[i],sdb1[i],man1[i],stan1[i]])
#    add1.append(stan[i]+sdb[i])
#    print(add1[i])    
#    test.append([mdb1[i],man1[i],mvb1[i],svb1[i]])
    test.append([mdb1[i],man1[i]])
#    test.append([mdb1[i],sdb1[i],man1[i],stan1[i]])
#timtest1=[]
#t=0;
#
#for i in range(stp,etp):
#    t = t + timediff(tmm1[i],tmm1[i+1])
#    timtest1.append(t)
#    test.append([timtest1[i-3001],vmm1y[i]])
#au=[]  
#------------------------------------------------------------------------------
#Custom Function for Finding Accuracy of Prediction:

au=clf.predict(test)
p=0;
cr1 = 0;
acc1 = [];
tt =[]
for i in range(tslen-1):
    p = p+1
    if au[i]==1:
        cr1 = cr1+1
    acc1.append(cr1/p)
    tt.append(p)
    
print('Accuracy Detecting Authorized User:',(cr1/(tslen-1))*100)
     
acc2 =[]   
cr = 0;
p1=0;
for i in range(tslen-1,(tslen-1)*2):
    p1=p1+1
    if au[i]==0:
        cr = cr+1
    acc2.append(cr/p1)
    
acc3 =100*np.add(np.array(acc1),np.array(acc2))/2
plt.plot(tt,acc3)
plt.xlabel('Number of Data points')
plt.ylabel('Testing Accuracy')
plt.title('Testing accuracy plot')
print('Accuracy Detecting Unauthorized User:',(cr/(tslen-1))*100)

print('Accuracy Detection :',((cr+cr1)/((tslen-1)*2))*100)
    
#%%

