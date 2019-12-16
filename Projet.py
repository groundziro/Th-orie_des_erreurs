import matplotlib.pyplot as plt
from scipy.stats import *
import math
import numpy as np
import copy
"""
Variables Globales
"""
fract = 1.96
snede = 1.39
alpha = 0.05
kolmo = 0.134
i=[]
Tables=[]
CPU1 = []
CPU2 = []
CPU3 = []

"""
Initialise toutes les tables globales
"""
def init():
    with open("./QI9_donnee.dat",'r') as f:
        for ligne in f.readlines():
            i.append(ligne.split())
    a = 0
    k = []
    for tab in i:
        for b in tab:
            k.append(float(b))
            a+=1
            if a == 4:
                Tables.append(k)
                k = []
                a=0
    index = 0
    TailleCPU1 = []
    TailleCPU2 = []
    TailleCPU3 = []
    for l in range(1000) :
        index+=1
        TailleCPU1.append(Tables[l][1])
        TailleCPU2.append(Tables[l][2])
        TailleCPU3.append(Tables[l][3])
        if index==100:
            index = 0
            CPU1.append(TailleCPU1)
            CPU2.append(TailleCPU2)
            CPU3.append(TailleCPU3)
            TailleCPU1 = []
            TailleCPU2 = []
            TailleCPU3 = []

def plot1():
    index = 10
    c1 = CPU1
    for test in c1:
        normaliser(test)
        x = np.arange(min(test),max(test)+1/float(len(test)))
        y = np.linspace(min(test),max(test),100)
        n, bins, patches = plt.hist(test,bins = x,color="blue",label="CPU",density=True,align='mid')
        plt.plot(y,((2*math.pi)**(0.5)*max(n)*norm.pdf(y,0,1)),label="norm",color='red')
        plt.title("Taille "+str(index))
        plt.legend()
        plt.savefig("Rapport/CPU1/Taille"+str(index)+".pdf")
        plt.clf()
        index+=10

def plot2():
    index = 10
    c1 = CPU2
    for test in c1:
        t = test[:]
        normaliser(t)
        x = np.arange(min(t),max(t)+1/float(len(test)))
        y = np.linspace(min(t),max(t),100)
        n, bins, patches = plt.hist(t,bins = x,color="blue",label="CPU",density=True,align='mid')
        plt.plot(y,((2*math.pi)**(0.5)*max(n)*norm.pdf(y,0,1)),label="norm",color='red')
        plt.title("Taille "+str(index))
        plt.legend()
        plt.savefig("Rapport/CPU2/Taille"+str(index)+".pdf")
        plt.clf()
        index+=10

def plot3():
    index = 10
    c1 = CPU3
    for test in c1:
        normaliser(test)
        x = np.arange(min(test),max(test)+1/float(len(test)))
        y = np.linspace(min(test),max(test),100)
        n, bins, patches = plt.hist(test,bins = x,color="blue",label="CPU",density=True,align='mid')
        plt.plot(y,((2*math.pi)**(0.5)*max(n)*norm.pdf(y,0,1)),label="norm",color='red')
        plt.title("Taille "+str(index))
        plt.legend()
        plt.savefig("Rapport/CPU3/Taille"+str(index)+".pdf")
        plt.clf()
        index+=10

def testMoyennes(l1,l2):
    avg1 = avgEstimate(l1)
    avg2 = avgEstimate(l2)
    s1 = sigmaEstimate(l1,avg1)**2
    s2 = sigmaEstimate(l2,avg2)**2
    a = abs(avg1-avg2)
    b = fract*math.sqrt(s1/100.0 + s2/100.0)
    return a<=b

def mieux(C1,C2):
    return avgEstimate(C1)<avgEstimate(C2)


def testVariances(l1,l2):
    n = 100.0/99.0
    sx = varianceEmpirique(l1,avgEstimate(l1))
    sy = varianceEmpirique(l2,avgEstimate(l2))
    s1 = max(sx,sy)
    s2 = min(sx,sy)
    T = s1/s2
    return T<snede

def varianceEmpirique(l,m):
    total = 0
    for i in l:
        total+= (i-m)**2
    return total/100.0

def Fn(l,i):
    cnt=0
    for j in l:
        if j<=i:
            cnt+=1
    return cnt

def Norm(i):
    return 100*norm.cdf(i)

def sigmaEstimate(l,m):
    total = 0
    for i in l:
        total+= (i-m)**2
    return math.sqrt(float(total)/100.0)

def avgEstimate(l):
    total = 0
    for i in l:
        total+=i
    return float(total)/float(len(l))


def normaliser(l):
    m = avgEstimate(l)
    s = sigmaEstimate(l,m)
    for i in range(100):
        l[i] = float(l[i]-m)/float(s)

"""
Cette fonction retourne true si, a une taille donnee, un CPU respecte une loi normale donc regarder si c'est une gaussienne
"""

def kolmogorov(taille):
    l = copy.copy(taille)
    normaliser(l)
    n = 100.0
    D = 0
    for x in l:
        inter = abs(Fn(l,x) - Norm(x))
        if inter > D :
            D = inter
    return (D/n)<kolmo

"""
Effectue les tests pour chaque taille de chaque CPU pour a la fin retourner le meilleur CPU entre C1 et C2

"""

def test(C1,C2):
    cntC1 = 0
    cntNull = 0
    cntC2 = 0
    meilleur = []
    for i in range(len(C1)):
        if kolmogorov(C1[i]) and kolmogorov(C2[i]):
            if testVariances(C1[i],C2[i]):
                if testMoyennes(C1[i],C2[i]):
                    cntNull+=1
                else:
                    if mieux(C1[i],C2[i]):
                        cntC1+=1
                    else:
                        cntC2+=1
            else:
                if mieux(C1[i],C2[i]):
                    cntC1+=1
                else:
                    cntC2+=1

    if cntC1<cntC2:
        meilleur = C2
    elif cntC2<cntC1:
        meilleur = C1
    else:
        meilleur = []
    return meilleur

if __name__ == "__main__":
    init()
    M1 = test(CPU1,CPU2)
    M2 = test(CPU1,CPU3)
    M3 = test(CPU2,CPU3)
    if M2 == M3:
        M = test(M1,M2)
        if M==M3:
            print("CPU3 est mieux")
        else:
            print("CPU1 est mieux")
    if M3==CPU2 :
        if M2 == [] and M1 == []:
            print("CPU2 est mieux")
        elif M2 != [] and M1==[]:
            print("CPU1 est hors course")
            print("--------------------")
            M = test(M3, M2)
            if M == M3:
                print("CPU2 est mieux")
            elif M == M2:
                if M2 == CPU1:
                    print("CPU1 est mieux")
                else:
                    print("CPU3 est mieux")
            else:
                if M2 == CPU1:
                    print("Aucun n'est mieux entre CPU1 et CPU2")
                else:
                    print("Aucun n'est mieux entre CPU2 et CPU3")
                print("-----------------------------------")
        elif M1!=[] and M2==[]:
            print("CPU2 est hors course")
            print("--------------------")
            M = test(M1,M3)
            if M == M3:
                print("CPU2 est mieux")
            elif M == M1:
                print("CPU1 est mieux")
            else:
                print("Aucun n'est mieux entre CPU1 et CPU2")
    if M3==CPU3:
        if M2 == [] and M1 == []:
            print("CPU3 est mieux")
        elif M2 != [] and M1==[]:
            M = test(M3, M2)
            if M==M3:
                print("CPU3 est mieux")
            else:
                print("CPU1 est mieux")
        elif M1 != [] and M2 ==[]:
            print("CPU2 est hors course")
            print("--------------------")
            M = test(M1,M3)
            if M==M3:
                print("CPU3 est mieux")
            elif M==M1:
                if M1 == CPU2:
                    print("CPU2 est mieux")
                else:
                    print("CPU1 est mieux")
            else:
                if M1 == CPU1:
                    print("Aucun n'est mieux entre CPU1 et CPU3")
                else:
                    print("Aucun n'est mieux entre CPU2 et CPU3")
                print("-----------------------------------")
    if M2 == CPU1:
        if M3 == [] and M1 == []:
            print("CPU1 est mieux")
        elif M3!= [] and M1==[]:
            print("CPU3 est hors course")
            print("--------------------")
            M = test(M3, M2)
            if M==M2:
                print("CPU1 est mieux")
            elif M==M3:
                if M3 == CPU2:
                    print("CPU2 est mieux")
                else:
                    print("CPU3 est mieux")
            else:
                if M3 == CPU2:
                    print("Aucun n'est mieux entre CPU1 et CPU2")
                else:
                    print("Aucun n'est mieux entre CPU1 et CPU3")
        elif M1 != [] and M2 ==[]:
            print("CPU2 est hors course")
            print("--------------------")
            M = test(M1,M2)
            if M==M2:
                print("CPU1 est mieux")
            elif M==M1:
                print("CPU2 est mieux")
            else:
                print("Aucun n'est mieux entre CPU1 et CPU2")
    if M2 == CPU3:
        if M3 == [] and M1 == []:
            print("CPU3 est mieux")
        elif M3 != [] and M1==[]:
            M = test(M3, M2)
            if M==M2:
                print("CPU3 est mieux")
            elif M==M3:
                print("CPU2 est mieux")
            else:
                print("Aucun n'est mieux entre CPU3 et CPU2")
        elif M1 != [] and M3 ==[]:

            M = test(M1,M3)
            if M==M3:
                print("CPU3 est mieux")
            elif M==M1:
                if M1 == CPU1:
                    print("CPU1 est mieux")
                else:
                    print("CPU2 est mieux")
            else:
                if M1 == CPU1:
                    print("Aucun n'est mieux entre CPU1 et CPU3")
                else:
                    print("Aucun n'est mieux entre CPU2 et CPU3")
    if M1 == CPU1:
        if M2==[] and M3 ==[]:
            print("CPU1 est mieux")
        elif M2!=[] and M3==[]:
            M = test(M1,M2)
            if M==M1:
                print("CPU1 est mieux")
            elif M==M2:
                print("CPU3 est mieux")
            else:
                print("Aucun n'est mieux entre CPU1 et CPU3")
        elif M3!=[] and M2==[]:
            print("CPU2 est hors course")
            print("--------------------")
            M = test(M1,M3)
            if M==M1:
                print("CPU1 est mieux")
            elif M==M3:
                if M3 == CPU2:
                    print("CPU2 est mieux")
                else:
                    print("CPU3 est mieux")
            else:
                print("Aucun n'est mieux entre CPU1 et CPU3")
    if M1 == CPU2:
        if M2==[] and M3==[]:
            print("CPU2 est mieux")
        elif M2!=[] and M3==[]:
            M = test(M1,M2)
            if M == M1:
                print("CPU2 est mieux")
            elif M == M2:
                if M2 == CPU3:
                    print("CPU3 est mieux")
                else:
                    print("CPU1 est mieux")
            else:
                if M2 == CPU1:
                    print("Aucun n'est mieux entre CPU1 et CPU2")
                else:
                    print("Aucun n'est mieux entre CPU2 et CPU3")
        elif M3!=[] and M2==[]:
            print("CPU2 est hors course")
            print("--------------------")
            M = test(M1,M3)
            if M == M1:
                print("CPU2 est mieux")
            elif M == M3:
                print("CPU3 est mieux")
            else:
                print("Aucun n'est mieux entre CPU2 et CPU3")
    print("FINISHED")
