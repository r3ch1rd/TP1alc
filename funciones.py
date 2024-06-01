import numpy as np
import networkx as nx
from scipy import linalg 
import time
import matplotlib.pyplot as plt

def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
    	line = f.readline()
    	i = int(line.split()[0]) - 1
    	j = int(line.split()[1]) - 1
    	W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W, print_ejes=True):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)
    
def descompLU(A):
    n = A.shape[0] # n = cantidad de filas, coincide con el orden de la matriz
    L = np.eye(n)  # Creamos la matriz del mismo tamaño de A pero la matriz identidad
    U = A.astype(float)          # La matriz U comienza siendo una copia de A

    for j in range(n):                               # Recorremos las columnas
        if U[j, j] != 0:                             # Si el pivote es distinto de 0
            for i in range(j+1, n):                  # Convertimos en ceros los elementos bajo el pivote
                L[i, j] = U[i, j]/U[j, j]            # Escribimos el coeficiente (elemento / pivote) que se utiliza en Gauss en la matriz L       
                U[i, :] = U[i, :] - L[i, j]*U[j, :]  # Convertimos ese elemento en 0 en la matriz U
        else:
            print('Error: Hubo un 0 en la diagonal (El pivote es 0 y no existe factorizacion LU)')
            return np.eye(n), A                      # Si el pivote es 0, se devuelve la identidad y la matriz A

    return L, U


def resolverLU(A, b):
    L, U = descompLU(A) # Reutilizamos la funcion del punto anterior
    y = linalg.solve_triangular(L, b, lower=True) # El vector 'y' tiene la solucion del sistema Ly=b . lower=True significa que la matriz es triangular inferior.
    x = linalg.solve_triangular(U, y, lower=False) # El vector 'x' tiene la solucion que buscamos (de Ax=b). lower=False significa que la matriz es triangular inferior.
    return x

def sumColumna(A, col):
    res = 0
    for i in range(A.shape[0]):
        res += A[i,col]
    return res

def calcularRanking(W, p):
    npages = W.shape[0]
    rnk = np.zeros(npages) # ind{k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 

    #Definimos D
    D = np.zeros(W.shape)
    for i in range(npages):
        if sumColumna(W, i) != 0:
            D[i,i] = 1/sumColumna(W, i)
        else:
            D[i,i] = 0
    #Definimos e
    e = np.ones(npages)
    
    #Definimos A
    A = np.identity(npages) - p * np.dot(W,D) 
    
    x = resolverLU(A, e)
    scr = (1 / np.sum(x))*x
    scr.round(12, scr) #Limita la cantidad de decimales a 6 porque daba errores

    #En base al scr, hacemos el rnk
    for r in range(1, npages+1):   #r es el numero en el ranking 
        for i in range(npages):    #i recorre todo el scr buscando el menor que los rankeados y mayor que los que no tienen ranking
            if rnk[i] == 0:
                mayor = True
                for j in range(npages):                 #Compara el scr[i] con todos los otros 
                    if scr[i] < scr[j] and rnk[j] == 0:   #Si scr[i] es menor que alguno de los no rankeados, entonces mayor = false
                        mayor = False
                if mayor == True:                       #si mayor == true, entonces rankea al scr[i] y busca el siguiente
                    rnk[i] = r
                    break
        
    return rnk, scr

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output


def calcularTiempoLinks(W): #Recibe una matriz W y retorna una dupla con; la cantidad de links y el tiempo que tardó
    start = time.time()
    calcularRanking(W, 0.5)
    end = time.time()

    links = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i,j] == 1:
                links += 1
            
    return (links,end-start)

def calcularTiempoPag(W): #Recibe una matriz W y retorna una dupla con; la cantidad de paginas y el tiempo que tardó
    start = time.time()
    calcularRanking(W, 0.5)
    end = time.time()
    return (W.shape[0],end-start)


