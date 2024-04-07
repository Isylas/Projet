#splines cubiques
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from numba import jit, f8

def splines_cubiques(x, y):
    '''
    Splines cubiques sans tridiagonalisation

    '''
    n=len(x)-1 # n+1 points
    # mat carrée :
    matFinale=np.zeros((4*(n),4*(n))) # n polynômes à trouver, 4n coefficients
    # matrice resultats:
    matRes=np.zeros((4*(n),1))
    #P(xi)=yi
    for i in range(n):
        for j in range(4*i, i*4+4, 1): # 4 coef dans un polynôme de deg 3.
            #Pi(xi-1)=yi-1
            matRes[i*2][0]=y[i]
            matFinale[i*2][j]=(x[i])**(j-4*i)
            
            #Pi(xi)=yi
            matRes[i*2+1][0]=y[i+1]
            matFinale[i*2+1][j]=(x[i+1])**(j-4*i)
            
            # ici on a 2n lignes
            
    # Continuité de la dérivée : donne n-1 lignes
    for i in range(0,n-1,1):
        for j in range(4):
            #Pi'(xi)=Pi+1'(xi)
            matFinale[2*(n+1)-2+i][j+i*4]=j*x[i+1]**(j-1)
            matFinale[2*(n+1)-2+i][j+i*4+4]=-j*x[i+1]**(j-1)
        # Continuité de la dérivée seconde : donne n-1 lignes
        matFinale[2*(n+1)-2+i+(n-1)][2+i*4]=2
        matFinale[2*(n+1)-2+i+(n-1)][3+i*4]=6*x[i+1]
        matFinale[2*(n+1)-2+i+(n-1)][6+i*4]=-2
        matFinale[2*(n+1)-2+i+(n-1)][7+i*4]=-6*x[i+1]
    # ici, on a 4n-2 lignes
    
    #Choix arbitraires : splines canoniques
    matFinale[4*n-2][3]=x[0]*6
    matFinale[4*n-2][2]=2
    matFinale[4*n-1][4*n-1]=6*x[n]
    matFinale[4*n-1][4*n-2]=2
    #On a 4n lignes. Resolution à faire:
    matFinale=matFinale.astype(float)
    return matFinale, matRes



def résolution(x, y):
    '''
    Calcule les matrices grâce à la fonction splines_cubiques
    puis résout le système qu'elle renvoie.

    '''
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
    return [2*b,6*a,0,0]
        

def dérivée_spline(x,y):  #renvoie la dérivée des splines sous la forme de matrice 3*(n-1)
    n=len(x)
    dérivée1_spline=[]
    matCoeff = résolution(x,y)
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
    for i in range (n-1):
        a=dérivée_seconde[4*i+1][0]
        b=dérivée_seconde[4*i][0]
        if -b/a >= x[i] and -b/a <= x[i+1]:
            res = -b/a
    return res


def affichage(X,Y,res):
    plt.figure()
    plt.scatter(X,Y,color='r')
    #print(len(X)-len(res))
    for i in range(len(X)-1):
        I = np.linspace(X[i], X[i+1], 10)
        F = res[(i*4)][0]+res[i*4+1][0]*I+res[i*4+2][0]*I**2+res[i*4+3][0]*I**3
        plt.plot(I,F, color='b')
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
    plt.figure()
    plt.scatter(X,Y,color='r')
    for i in range(len(res)-1):
        I = np.linspace(X[i], X[i+1], 10)
        F = ((res[i]*(X[i+1]-I)**3)/(6*pas)+res[i+1]*(I-X[i])**3/(6*pas)+(I-X[i])*(Y[1+i]-(pas*pas*res[1+i])/6)/pas+(X[1+i]-I)*(Y[i]-(pas*pas*res[i])/6)/pas)
        plt.plot(I,F, color='b')
    plt.show()
    

def résolutionTridiagoNp(x, y):
    '''
    Résout le système tridiagonal en utilisant le module de np.
    '''
    matFinale, matRes = tridiagonale(x, y)
    matFinale_inv = np.linalg.inv(matFinale)
    matCoeff = np.dot(matFinale_inv, matRes)
    return matCoeff


## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
# Pour gagner du temps, n'a pas été ecrit par nos soins.
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


# --------------------- Tests


x=[0.4,0.8,1.2,1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6,6,6.4,6.8,7.2,7.6,8,8.4,8.8,9.2,9.6,10,10.4,10.8,11.2,11.6,12,12.4,12.8,13.2,13.6,14,14.4,14.8,15.2,15.6,16,16.4,16.8,17.2,17.6,18,18.4,18.8,19.2,19.6,20]
y=[1.42265,1.4276960317453,1.43312785633646,1.43899754166588,1.44536020444814,1.45228060739777,1.45983536352571,1.46811577308034,1.47723151530769,1.48731551094278,1.49853041197118,1.51107739025696,1.52520823252366,1.5412422861543,1.55959068121755,1.58079174169147,1.60556409358411,1.63488867277774,1.67013969426102,1.7133021834278,1.76735038908997,1.83694347140276,1.92979307030677,2.05958216452858,2.25286268280524,2.56749349899594,3.14835947756378,4.37874149918824,6.59218667954964,8.35619449019234,9.19109346720768,9.61086747893003,9.85423465523273,10.011159437778,10.120202300835,10.2001833459588,10.2612746963633,10.3094246545258,10.3483329786724,10.380417316863,10.4073224250955,10.4302052765007,10.4499029579775,10.4670358385307,10.4820733065754,10.4953769887689,10.5072299496132,10.5178568977251,10.5274384684669,10.5361215122375]


#Sans tridiago, ;
# carr, res = tridiagonale(x, y)
# affichage(x, y, résolution(x, y))

# Avec tridiago ;
#a, b, c, d = tridiagonaleabcd(x, y)
#affichageTridiago(x, y, (résolutionTridiagoNp(x, y)))  # utilise inverse de np
# print(TDMAsolver(a, b, c, d)) # matrices du solveur tridiago fait maison
#affichageTridiago(x, y, TDMAsolver(a, b, c, d)) # utilise inverse tridiago fait maison 

#%%

# Simulations d'erreurs

def simuler_erreur_etalonnage():
    delta_ph_4=np.random.uniform(0.016, 0.021)
    delta_ph_7=np.random.uniform(-0.002, 0.002)
    eps_4=np.random.normal(0, 0.2)
    eps_7=np.random.normal(0, 0.2)
    A=(4-7)/(delta_ph_4-delta_ph_7) #éronné
    a=(4+eps_4-7-eps_7)/(delta_ph_4-delta_ph_7)
    C=4-A*delta_ph_4
    c=4+eps_4-a*(delta_ph_4)
    return A, C, a, c

def simuler_erreur_aleatoire(y):
    U=[]
    for x in range(len(y)):
        U.append(np.random.normal(0, 0.1))  #0.1 arbitraire
    return U

def simuler_err_aberrante(y):
    y_err_aberrante=y.copy()
    for x in range(len(y_err_aberrante)):
        test=np.random.randint(1, 41) # borne du haut non incluse
        if test==3:
            modif=np.random.uniform(2, 3)
            signe=np.random.randint(0, 2)
            if signe==0:
                y_err_aberrante[x]-=modif
            else:
                y_err_aberrante[x]+=modif
                
    #le pH-mètre ne peut pas afficher un pH < 0 ou > 14.
    
    for x in range(len(y)):
        if y_err_aberrante[x] < 0:
            y_err_aberrante[x] = 0
        elif y_err_aberrante[x] > 14:
            y_err_aberrante[x] = 14
    return y_err_aberrante
        
def sim_erreurs(y):
    y_er = simuler_err_aberrante(y)
    A, C, a, c = simuler_erreur_etalonnage()
    for i in range(len(y_er)):
        deltaE=(y_er[i]-c)/a # On cherche à quel deltaE notre pH correspond (ce qui a été mesuré)
        y_er[i] += (A-a) * deltaE+ (C-c) # Puis on applique la fonction d'erreur 
    U = simuler_erreur_aleatoire(y)
    return [m+p for m, p in zip(U, y_er)]


# y_erronee=sim_erreurs(y)
# aber=simuler_err_aberrante(y)
# plt.scatter(x, y, label='Points sans erreurs')
# plt.scatter(x, y_erronee, label='Points avec erreurs')


# plt.xlabel('Volume (mL)', fontsize=18)
# plt.ylabel('pH', fontsize=18)

# # Ajuste la taille de la police des nombres sur les axes x et y
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Ajoute une légende
# plt.legend(fontsize=20)

# plt.show()


