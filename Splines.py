#splines cubiques
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from numba import jit, f8

def splines_cubiques(x, y):
    '''
    Splines cubiques sans tridiagonalisation

    '''
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
    '''
    Calcule les matrices grâce à la fonction splines_cubiques
    puis résout le système qu'elle renvoie.

    '''
    print(x, y)
    matFinale, matRes = splines_cubiques(x, y)
    matFinale_inv = np.linalg.inv(matFinale)
    matCoeff = np.dot(matFinale_inv, matRes)
    return matCoeff

        
def dérivée_polynome (poly):
    '''
    
    dérive un polynome de degré 3 (4 coeff)
    
    '''
    
    a=poly[3]
    b=poly[2]
    c=poly[1]
    return [c, 2*b, 3*a,0]


def dérivée_seconde(poly):
    a=poly[3]
    b=poly[2]
    return([2*b,6*a,0,0])
        

def dérivée_spline(x,y):  #renvoie la dérivée des splines sous la forme de matrice 3*(n-1)
    n=len(x)
    dérivée1_spline=[]
    matCoeff = résolution(x,y)
    print(matCoeff)
    for i in range (n-1):
        dérivée=[]
        poly = [matCoeff[4*i][0],matCoeff[4*i+1][0],matCoeff[4*i+2][0],matCoeff[4*i+3][0]]
        dérivée=dérivée_polynome (poly)
        for j in range(4):
            dérivée1_spline.append([dérivée[j]])
    return dérivée1_spline


def dérivée_seconde_spline(x,y):
    n=len(x)
    dérivée_2_spline=[]
    matCoeff = résolution(x,y)
    for i in range (n-1):
        dérivée=[]
        poly = [matCoeff[4*i][0],matCoeff[4*i+1][0],matCoeff[4*i+2][0],matCoeff[4*i+3][0]]
        dérivée=dérivée_seconde(poly)
        for j in range(4):
            dérivée_2_spline.append([dérivée[j]])
    return dérivée_2_spline


def equilibre(x,y):
    '''
    
    recherche le point d'équilibre d'un jeu de données donné, utilisant 
    les splines cubiques sans tridiagonalisation.
    
    '''
    n=len(x)
    res = None
    dérivée_seconde= dérivée_seconde_spline(x,y)
    print(dérivée_seconde)
    for i in range (n-1):
        a=dérivée_seconde[4*i+1][0]
        b=dérivée_seconde[4*i][0]
        print(b)
        if -b/a >= x[i] and -b/a <= x[i+1]:
            res = -b/a
    return res


def affichage(X,Y,res):
    x=symbols('x')
    
    P= plot(res[0][0]+res[1][0]*x+res[2][0]*x**2+res[3][0]*x**3,(x,X[0],X[1]), show=False,line_color='b')
    for i in range(1, np.shape(res)[0]//4, 1):
        p=plot(res[(i*4)][0]+res[i*4+1][0]*x+res[i*4+2][0]*x**2+res[i*4+3][0]*x**3,(x,X[i],X[i+1]), show=False,line_color='b')
        P.extend(p)
    P.show()
    plt.scatter(X,Y)
    plt.show()


#-------------------------- Splines cubiques avec tridiagonalisation

def tridiagonale(X, Y): #renvoie la matrice carré, plus utilisé
    n=len(X)
    assert len(X)==len(Y)
    matRes=np.zeros((n-2, 1))
    pas=X[1]-X[0]
    matCar=np.zeros((n-2,n-2))
    for i in range(n-2):
        matCar[i][i]=2*pas/3
        matRes[i][0]=(Y[i+2]-2*Y[i+1]+Y[i])/pas
    for i in range(n-3):
        matCar[i][i+1]=pas/6
        matCar[i+1][i]=pas/6
    return matCar, matRes


def tridiagonaleabcd(X, Y): # renvoie a, b, c, d (adapté fonction d'au dessus)
    n=len(X)
    assert len(X)==len(Y)
    matRes=np.zeros((n-2, 1))
    pas=X[1]-X[0]
    a=np.full((n-3, 1), pas/6) #coef diag dessous
    b=np.full((n-2,1), 2*pas/3) # coef diag
    c=np.full((n-3, 1), pas/6) # coefs diag dessus
    for i in range(n-2):
        matRes[i][0]=(Y[i+2]-2*Y[i+1]+Y[i])/pas
    return a, b, c, matRes



def affichageTridiago(X,Y,res):
    res=np.append(res, 0)
    res=np.insert(res, 0, 0) # Ajout de M0 et de Mn
    assert len(X)==np.shape(res)[0]
    pas=X[1]-X[0]
    x=symbols('x')
    P= plot(res[0]*(X[1]-x)**3/(6*pas)+res[1]*(x-X[0])**3/(6*pas)+(x-X[0])*(Y[1]-(pas*pas*res[1])/6)/pas+(X[1]-x)*(Y[0]-(pas*pas*res[0])/6)/pas, (x,X[0],X[1]), show=False,line_color='b')
    for i in range(1, len(res)-1):
        p=plot((res[i]*(X[i+1]-x)**3)/(6*pas)+res[i+1]*(x-X[i])**3/(6*pas)+(x-X[i])*(Y[1+i]-(pas*pas*res[1+i])/6)/pas+(X[1+i]-x)*(Y[i]-(pas*pas*res[i])/6)/pas, (x,X[i],X[i+1]), show=False,line_color='b')
        P.extend(p)
    P.show()
    plt.scatter(X,Y)
    plt.show()


def résolutionTridiagoNp(x, y):
    matFinale, matRes = tridiagonale(x, y)
    matFinale_inv = np.linalg.inv(matFinale)
    matCoeff = np.dot(matFinale_inv, matRes)
    return matCoeff


## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    
    trouvé ici : https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


# -------------------- Tests


x=[0.4,0.8,1.2,1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6,6,6.4,6.8,7.2,7.6,8,8.4,8.8,9.2,9.6,10,10.4,10.8,11.2,11.6,12,12.4,12.8,13.2,13.6,14,14.4,14.8,15.2,15.6,16,16.4,16.8,17.2,17.6,18,18.4,18.8,19.2,19.6,20]
y=[1.42265,1.4276960317453,1.43312785633646,1.43899754166588,1.44536020444814,1.45228060739777,1.45983536352571,1.46811577308034,1.47723151530769,1.48731551094278,1.49853041197118,1.51107739025696,1.52520823252366,1.5412422861543,1.55959068121755,1.58079174169147,1.60556409358411,1.63488867277774,1.67013969426102,1.7133021834278,1.76735038908997,1.83694347140276,1.92979307030677,2.05958216452858,2.25286268280524,2.56749349899594,3.14835947756378,4.37874149918824,6.59218667954964,8.35619449019234,9.19109346720768,9.61086747893003,9.85423465523273,10.011159437778,10.120202300835,10.2001833459588,10.2612746963633,10.3094246545258,10.3483329786724,10.380417316863,10.4073224250955,10.4302052765007,10.4499029579775,10.4670358385307,10.4820733065754,10.4953769887689,10.5072299496132,10.5178568977251,10.5274384684669,10.5361215122375]


# Sans tridiago, ;
carr, res = tridiagonale(x, y)
# affichage(x, y, résolution(x, y))

# Avec tridiago ;
a, b, c, d = tridiagonaleabcd(x, y)
affichageTridiago(x, y, (résolutionTridiagoNp(x, y)))  # utilise inverse de np
# print(TDMAsolver(a, b, c, d)) # matrices du solveur tridiago fait maison
affichageTridiago(x, y, TDMAsolver(a, b, c, d)) # utilise inverse tridiago fait maison 

