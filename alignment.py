#####################################################
# Licence Creative Commons - Attribution 3.0 France #
# https://creativecommons.org/licenses/by/3.0/fr/   #
#####################################################

###########################
# Importation des modules #
###########################

import numpy as np
import glob
from cv2 import *
from PIL import Image
from time import time
from datetime import timedelta
from math import *
from random import *
from scipy.spatial import Delaunay

############################
# Définition des fonctions #
############################

# Barre de chargement
def bar(val, maxval):
    digits = ceil(log10(maxval))
    ratio = val/maxval
    
    # Affiche la barre de chargement
    print('▕' + '█'*round(50*ratio) + ' '*round(50 - 50*ratio) + '▏', end='')
    print('     ', end='')
    
    # Affiche le pourcentage d'images chargées
    print(str(round(100*ratio)).zfill(3) + ' %', end='')
    print('     ', end='')
    
    # Affiche le nombre d'image chargées par rapport au nombre total d'images
    print(str(val).zfill(digits), 'sur', str(maxval).zfill(digits), end='')
    
    if val < maxval:
        print('\r', end='') # Retour au début de la ligne pour la mettre à jour
    else:
        print('') # Nouvelle ligne car fin du chargement

# Fonction de chargement des images
def load(root):
    imgs = [] # Liste des images
    nb = len(glob.glob(root)) # Nombre d'images trouvées
    if nb == 0:
        print('Aucune image trouvée')
        return # On quitte la fonction

    # Initialisation
    print(str(nb).zfill(3), 'images trouvées')
    print('')
    bar(0, nb)
    i = 0
    start = time()
    
    # Chargement des images
    for path in glob.glob(root): # Pour chaque image
        # Charge l'image sous forme de tableau
        image = Image.open(path)
        imgArr = np.asarray(image)
        imgArr = np.uint8(image)
        imgs.append(imgArr) # Ajoute l'image à la liste 'imgs'
        i += 1 # Incrémente le nombre d'images chargées
        bar(i, nb) # Met à jour la barre de chargement
        
    # Affiche le temps écoulé
    end = time()
    print('')
    print(timedelta(seconds=(end - start)))
    return imgs # Retourne la liste des images chargées

# Conversion en nuances de gris
def rgbToGray(array):
    gray = cvtColor(array, COLOR_RGB2GRAY) # Conversion RGB en nuances de gris
    return gray

# Lissage
def smoothTwice(array): # Double lissage
    median = medianBlur(array, 3) # Filtre médian
    bilateral = bilateralFilter(median, 3, 3, 3) # Filtre bilatéral
    return bilateral

# Binarisation selon la luminosité des pixels
def binarise(array):
    array[array < 128] = 0 # Toute valeur comprise entre 0 et 127 devient 0
    array[array > 0] = 255 # Toute valeur comprise entre 128 et 255 devient 255
    return array

# Extraction des points d'intérêt
def getCentroids(array): # Extraction du centre des régions connexes
    # Extraction des régions connexes et de leur centre
    nlabels, labels, stats, centroids = connectedComponentsWithStats(array)
    return centroids

def pointsToTriangles(points):
    # Détermine la triangulation de Delaunay des points d'intérêt
    delaunay = Delaunay(points)
    triangles = []
    
    for triangle in delaunay.simplices: # Pour chaque triangle ABC
        i, j, k = triangle # Index de A, de B, de C dans le tableau 'points'
        pointA = np.array(points[i]) # Coordonnées de A
        pointB = np.array(points[j]) # Coordonnées de B
        pointC = np.array(points[k]) # Coordonnées de C
        
        # Vecteurs représentant les trois côtés
        edgeA = pointB - pointC # Côté [BC] opposé au point A
        edgeB = pointA - pointC # Côté [AC] opposé au point B
        edgeC = pointA - pointB # Côté [AB] opposé au point C
        
        # Calcule les longueurs des trois côtés
        distA = hypot(edgeA[0], edgeA[1]) # Longueur BC
        distB = hypot(edgeB[0], edgeB[1]) # Longueur AC
        distC = hypot(edgeC[0], edgeC[1]) # Longueur AB
        
        # Stocke dans le désordre les points et la longueur des côtés opposés
        triangle = [(pointA, distA), (pointB, distB), (pointC, distC)]
        # Range les points et les distances associées par distance croissante
        triangle.sort(key=lambda pair: pair[1])
        [(point1, dist1), (point2, dist2), (point3, dist3)] = triangle
        
        # Calcule le descripteur D du triangle
        Desc = (dist2/dist1, dist3/dist1)
        
        # Si le triangle est trop régulier/particulier
        if Desc[0] < 1.1 or dist3/dist2 < 1.1:
            continue
            
        # Stocke D et les points rangés par longueur croissante du côté opposé
        triangle = (Desc, point1, point2, point3)
        triangles.append(triangle)
        
    return triangles

def getMatches(srcTriangles, dstTriangles):
    matches = []
    
    for srcTriangle in srcTriangles: # Pour chaque triangle de référence
        srcD = np.array(srcTriangle[0]) # Extraire son descripteur
        ref = hypot(srcD[0], srcD[1])
        
        for dstTriangle in dstTriangles: # Pour chaque triangle cible
            dstD = np.array(dstTriangle[0]) # Extraire son descripteur
            diff = dstD - srcD
            score = hypot(diff[0], diff[1]) / ref
            if score <= 0.01: # Si les triangles sont jugés similaires
                # Stocker la correspondance
                matches.append((score, srcTriangle[1:], dstTriangle[1:]))

    matches.sort(key=lambda match: match[0])
    return matches

#########################
# Chargement des images #
#########################

# Charge les fichiers qui se situent dans le dossier 'images'
images = load('photographies/*')

##############################
# Traitement de chaque image #
##############################

# Points d'intérêt de chaque image
allTriangles = []

for imgArr in images: # Pour chaque image
    grayscale = rgbToGray(imgArr) # Conversion en nuances de gris
    smooth = smoothTwice(grayscale) # Lissage
    binary = binarise(smooth) # Binarisation selon la luminosité des pixels
    centroids = getCentroids(binary) # Extraction des points d'intérêt
    centroids = centroids[1:] # Ignore le fond noir d'index 0
    triangles = pointsToTriangles(list(centroids))
    allTriangles.append(triangles)

srcTriangles, allDstTriangles = allTriangles[0], allTriangles[1:]
srcLimit = len(srcTriangles)
(height, width, channels) = images[0].shape
aligned = [images[0]]

##############################
# Comparaison de deux images #
##############################

# Initialisation
nb = len(allDstTriangles)
print('')
print(str(nb).zfill(3), 'images à aligner')
print('')
bar(0, nb)
start = time()

for i, dstTriangles in enumerate(allDstTriangles, 1): # Pour chaque image cible
    # Extrait les meilleures correspondances
    # Il ne peut pas y avoir plus de correspondances que de triangles
    limit = np.array((srcLimit, len(dstTriangles))).min()
    matches = getMatches(srcTriangles, dstTriangles)[:limit]
    
    # Listes de triangles
    srcTrianglesM = [match[1] for match in matches]
    dstTrianglesM = [match[2] for match in matches]
    
    # Listes de points
    srcPoints = [point for triangle in srcTrianglesM for point in triangle]
    dstPoints = [point for triangle in dstTrianglesM for point in triangle]
    
    # Conversion en tableaux numpy ('ndarray')
    srcPoints = np.array(srcPoints)
    dstPoints = np.array(dstPoints)
    
    H, mask = findHomography(dstPoints, srcPoints, RANSAC)
    
    # Aligne l'image cible avec l'image de référence
    alignedDst = warpPerspective(images[i], H, (width, height))
    aligned.append(alignedDst)
    bar(i, nb) # Met à jour la barre de progression
        
# Affiche le temps écoulé
end = time()
print('')
print(timedelta(seconds=(end - start)))