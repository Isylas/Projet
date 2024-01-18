#splines cubiques
import numpy as np

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
    matFinale[4*(n-1)-2][2]=2
    matFinale[4*(n-1)-1][3*n-1]=x[n-1]*6
    matFinale[4*(n-1)-1][3*n-2]=2
    #On a 4(n-1) lignes. Resolution à faire:
    matFinale=matFinale.astype(int)
    return matFinale, matRes


def résolution (x,y):  #renvoie les coeff des ploynomes sous forme de matrice colonne (d1,c1,..., an-1)
    matFinale = splines_cubiques(x,y)[0]
    matRes = splines_cubiques(x, y)[1]
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
    

x=[1,3,5,8, 10]
y=[2,3,9,10, 11]
print (résolution(x,y))

print(dérivée_seconde_spline(x,y))
