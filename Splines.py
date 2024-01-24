#splines cubiques
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def splines_cubiques(x, y):
    n=len(x)
    # mat carrée :
    matFinale=np.zeros((4*(n-1),4*(n-1))) #n-1 polynômes à trouver
    # matrice resultats:
    matRes=np.zeros((4*(n-1),1))
    #P(xi)=yi
    for i in range(n-1):
        for j in range(4*i,i*4+4,1): #4 coef dans un polynôme de deg 3.
            #P(i)=xi
            matRes[i*2][0]=y[i]
            matFinale[i*2][j]=(x[i])**(j-4*i)
            #P(i+1)=xi+1
            matRes[i*2+1][0]=y[i+1]
            matFinale[i*2+1][j]=(x[i+1])**(j-4*i)
            #ici on a 2n-2 lignes
    #Continuité de la dérivée
    for i in range(0,n-2,1):
        for j in range(4):
            matFinale[2*n-2+i][j+i*4]=j*x[i+1]**(j-1)
            matFinale[2*n-2+i][j+i*4+4]=-j*x[i+1]**(j-1)
        matFinale[2*n-2+i+(n-2)][2+i*4]=2
        matFinale[2*n-2+i+(n-2)][3+i*4]=6*x[i+1]
        matFinale[2*n-2+i+(n-2)][6+i*4]=-2
        matFinale[2*n-2+i+(n-2)][7+i*4]=-6*x[i+1]
    #on a 3n-4 lignes
    #Continuité de la dérivée seconde
    
    #on a 2n+2 lignes
    #Choix arbitraires
    matFinale[4*(n-1)-2][3]=x[0]*6
    matFinale[4*(n-1)-2][2]=1
    matFinale[4*(n-1)-1][4*(n-1)-1]=6*x[n-1]
    matFinale[4*(n-1)-1][4*(n-1)-2]=1
    #On a 4(n-1) lignes. Resolution à faire:
    matFinale=matFinale.astype(float)
    return matFinale, matRes


def résolution(x, y):
    matFinale, matRes = splines_cubiques(x, y)
    print("Matrice finale avant inversion:")
    print(matFinale)
    matFinale_inv = np.linalg.inv(matFinale)
    matCoeff = np.dot(matFinale_inv, matRes)
    return matCoeff

        
def dérivée_polynome (poly): # dérive un polynome de degré 3 (4 coeff)
    a=poly[3]
    b=poly[2]
    c=poly[1]
    return [c, 2*b, 3*a]
        
def dérivée_spline(x,y):  #renvoie la dérivée des splines sous la forme de matrice 3*(n-1)
    n=len(x)
    dérivée_spline=[]
    matCoeff = résolution(x,y)
    for i in range (n-1):
        poly = [matCoeff[4*i][0],matCoeff[4*i+1][0],matCoeff[4*i+2][0],matCoeff[4*i+3][0]]
        dérivée_spline.append(dérivée_polynome (poly))    
    return dérivée_spline

def dérivée_seconde_spline(x,y):
    n=len(x)
    dérivée_seconde_spline=[]
    dérivée_première=dérivée_spline(x,y)
    for i in range(n-1):
        a=dérivée_première[i][2]
        b=dérivée_première[i][1]
        dérivée_seconde_spline.append([b,2*a])
    return dérivée_seconde_spline

def equilibre(x,y):

    n=len(x)
    res = None
    dérivée_seconde= dérivée_seconde_spline(x,y)
    for i in range (n-1):
        a=dérivée_seconde[i][1]
        b= dérivée_seconde[i][0]
        if -b/a >= x[i] and -b/a <= x[i+1]:
            res = -b/a
    return res



x=[0.4,0.8,1.2,1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6,6,6.4,6.8,7.2,7.6,8,8.4,8.8,9.2,9.6,10,10.4,10.8,11.2,11.6,12,12.4,12.8,13.2,13.6,14,14.4,14.8,15.2,15.6,16,16.4,16.8,17.2,17.6,18,18.4,18.8,19.2,19.6,20]
y=[1.42265,1.4276960317453,1.43312785633646,1.43899754166588,1.44536020444814,1.45228060739777,1.45983536352571,1.46811577308034,1.47723151530769,1.48731551094278,1.49853041197118,1.51107739025696,1.52520823252366,1.5412422861543,1.55959068121755,1.58079174169147,1.60556409358411,1.63488867277774,1.67013969426102,1.7133021834278,1.76735038908997,1.83694347140276,1.92979307030677,2.05958216452858,2.25286268280524,2.56749349899594,3.14835947756378,4.37874149918824,6.59218667954964,8.35619449019234,9.19109346720768,9.61086747893003,9.85423465523273,10.011159437778,10.120202300835,10.2001833459588,10.2612746963633,10.3094246545258,10.3483329786724,10.380417316863,10.4073224250955,10.4302052765007,10.4499029579775,10.4670358385307,10.4820733065754,10.4953769887689,10.5072299496132,10.5178568977251,10.5274384684669,10.5361215122375]
# x=[1,2,3,4,5,6]
# y=[1,3,6,9,1,-2]

print (résolution(x,y))


print(dérivée_seconde_spline(x,y))


def affichage(X,Y,res):
    x=symbols('x')
    
    P= plot(res[0][0]+res[1][0]*x+res[2][0]*x**2+res[3][0]*x**3,(x,X[0],X[1]), show=False,line_color='b')
    for i in range(1, np.shape(res)[0]//4, 1):
        print(res[(i*4)][0]+res[i*4+1][0]*x+res[i*4+2][0]*x**2+res[i*4+3][0]*x**3)
        p=plot(res[(i*4)][0]+res[i*4+1][0]*x+res[i*4+2][0]*x**2+res[i*4+3][0]*x**3,(x,X[i],X[i+1]), show=False,line_color='b')
        P.extend(p)
    P.show()
    plt.scatter(X,Y)
    plt.show()



res=résolution (x,y)
affichage(x,y,res)

