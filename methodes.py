import numpy as np
import cv2
import math
import copy
def get_skeleton(img, Show_Me=False):
    mask = copy.deepcopy(img)
    from skimage.morphology import skeletonize
    skel = (skeletonize(mask // 255) * 255).astype(np.uint8)
    if Show_Me:
        cv2.imshow("skeleton", skel)
        cv2.destroyAllWindows()
    return skel

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)
    # apply the convolution
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)
    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out = np.where(filtered == 11)
    array = np.array(out)
    coordinates = array.transpose()
    return coordinates

def get_adjacent_closed_skel_cnt(skel_cnt_i, ShowMe=False):
    """
    Algorithme d'amincissement simple qui utilise des éléments structurants comme il est décrit ici kernel
    Les éléments de structuration sont appliqués à l'aide de l'opération 'hit-miss' jusqu'à ce qu'aucun autre changement ne soit observé. Les éléments de structuration sont décrits à l'aide du code ci-dessous:

    choix du kernel:
    Le noyau 'Line Junctions' est généralement utilisé à deux fins.
    Compter le nombre de jonctions de lignes dans une image et travailler ainsi pour obtenir le nombre de segments de lignes du squelette.
    Déconnectez tous les segments de ligne les uns des autres.
    Notez toutefois qu'aux points de jonction "T" et "+" de l'image entré (skel_cn_i), le noyau de la jonction "Y" correspond aux points situés à un pixel d'un pixel de l'intersection réelle. À cause de cela, le nombre de jonctions peut ne pas être exactement comme prévu, en particulier au «+» où quatre correspondances ont été trouvées plutôt que les deux requises pour un comptage de jonctions. La prudence est recommandée.
    """
    # le noyau: si = 0 (noir) alors ne pas prendre si = 1 (blanc) alors on #prend
    kernel_0 = np.array((
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0]), dtype="int")

    kernel_1 = np.array((
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0]), dtype="int")

    kernel_2 = np.array((
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0]), dtype="int")

    kernel_3 = np.array((
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]), dtype="int")

    kernel_4 = np.array((
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0]), dtype="int")

    kernel_5 = np.array((
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 0]), dtype="int")

    kernel_6 = np.array((
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0]), dtype="int")

    kernel_7 = np.array((
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]), dtype="int")
    kernel = np.array((kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7))
    # comme on peut le voir on a pris les noyaux d'une façon à prendre les #lignes et non pas les jointures
    output_image = np.zeros(skel_cnt_i.shape)  # dtype=np.uint8
    for i in np.arange(8):
        """
        Erode image skel_cnt_i with structuring element kernel (where there is 1).
        Erode the complement of image skel_cnt_i (not skel_cnt_i) with structuring element kernel (where there is 0).
        AND results from step 1 and step 2.
        """
        out = cv2.morphologyEx(skel_cnt_i, cv2.MORPH_HITMISS, kernel[i, :, :])
        # hit et miss out est l'element retourné par cette méthode
        """
        by eroding the pixel with the structuring element kerenl[i,:,:] if it hitt (match) then we got 255 + the precedent 
        intincity pixel : output_image=output_image+out
        n, m = output_image.shape
        for ti in range(n):
            for tj in range(m):
                if output_image[ti][tj]!=0 and output_image[ti][tj]!=255 and output_image[ti][tj]!=510.0:
                    #if the pixel is surrounded by 8 pixel white -> output_image[ti][tj] == 2040 in the end of the treatement
                    print(output_image[ti][tj])

        """
        output_image = output_image + out
    # on fait une convention de type de float64 vers uin8 :
    # output_image contient des valeurs de pixls suivantes:
    # 255*8=2040,255*7=1785,255*6=1530..., 510,255
    #  une étape de standarisation est nécessaire pour eviter les overflow

    # output_image contient maintenant:#8,7,6,5,4...,1

    # output a un dégradé de transparence i.e:
    # { 1... 248} lines+junction en transparence + -> -
    # {249...254} the junctions en transparence + -> -  un pixel 254 correspond par exmple à 8 pixel voisin 255
    # un pixel de output maintenant = 255 * nombre de pixel voisin blanc / 255
    # concatener les eléments retourné avec le output_image
    info = np.iinfo(skel_cnt_i.dtype)
    #print(info.max)  # max=255
    # data = output_image/ info.max#dtype=float64
    data = output_image  # data * 255
    img = data.astype(np.uint8)  # img= img%256+1
    junction_5_255 = skel_cnt_i.copy() - img.copy()
    thresh_outup = cv2.threshold(junction_5_255.copy(), 1, 255, cv2.THRESH_BINARY)[1]
    # get the coordiantes of junction pixels:
    indices = np.where(thresh_outup == [255])
    # print(type(indices))
    array = np.array(indices)
    # print("shape=")
    coordinates = array.transpose()
    if ShowMe:
        cv2.imshow("skeleton mask cnt_ i", skel_cnt_i)
        cv2.imshow("junction_1-255", thresh_outup)
        cv2.waitKey(0)
    return cv2.findContours(thresh_outup.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]#, coordinates  # , np.where(output_image == 1)#thresh_outup,

def get_skel(img):
    mask = copy.deepcopy(img)
    from skimage.morphology import skeletonize
    skel = (skeletonize(mask // 255) * 255).astype(np.uint8)
    return skel

def methode3(cnt, i, pretraitement_1, internal_region,
             Show_me=False):
    # draw the region to get it s skeleton
    #print(pretraitement_1)

    mask = np.zeros(pretraitement_1.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt[i]], -1, 255, -1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(internal_region))
    skel = get_skel(mask)
    sek = cv2.findContours(skel.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    mm =  np.zeros(pretraitement_1.shape, dtype=np.uint8)
    for i in range(len(sek)):
        cv2.drawContours(mm, [sek[i]],-1,255,-1)
    cnt_skel_adjacent  = get_adjacent_closed_skel_cnt(mm)
    #face_candidates, correspondance = update_correspondance_face_candidate_structure(cnt_skel_adjacent, i,
                                                                               #      face_candidates, correspondance
    return cnt_skel_adjacent  # ,end_points#cnt_sub_cnt_non_face

# ==============================================================================
# cette methode est appelé dans la méthode qui suit
def compare_angle(cnt, i, j, Angle, min, max, pretraitement_1_copy_):
    # boll pour indiquer si la condition de l’angle est vérifier <-85et>-90
    boll = False
    shape = (pretraitement_1_copy_.shape[0], pretraitement_1_copy_.shape[1])
    mask2 = np.zeros(shape, dtype=np.uint8)
    # =====créer un contour vide pour le remplir avec les contours splited===
    cnt_splitted, hei = cv2.findContours(np.zeros(pretraitement_1_copy_.shape, dtype=np.uint8),
                                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if Angle <= max and Angle >= min:
        mask1 = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask1, [cnt[i]], -1, 255, -1)
        mask2 = cv2.subtract(pretraitement_1_copy_, mask1)
        # ========================================================================
        mask0 = np.zeros(pretraitement_1_copy_.shape, dtype=np.uint8)
        cv2.drawContours(mask0, [cnt[i]], -1, 255, -1)
        cv2.line(mask0, (cnt[i][j][0][0], 0),
                 (cnt[i][j][0][0], pretraitement_1_copy_.shape[0]), 0, 1)
        cnt_splitted,hei = cv2.findContours(mask0.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # =======================================================================
        mask2 = cv2.bitwise_or(mask2, mask0)
        for j in range(len(cnt_splitted)):
            mm = np.zeros(pretraitement_1_copy_.shape, dtype=np.uint8)
            cv2.drawContours(mm,[cnt_splitted[j]],-1,255,-1)
            #cv2.imshow(str(j),mm)
        #cv2.waitKey(0)
        boll = True
    return cnt_splitted, mask2, boll

def getAngle(a,b,c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang

def methode2(cnt, i, pretraitement_1, Show_me=False):
    pretraitement_1_copy = pretraitement_1.copy()
    cnt_new = []
    for j in range(4, len(cnt[i]) - 4):
        a = j-4
        b=j
        c=j+4
        Angle =getAngle(cnt[i][a][0], cnt[i][b][0], cnt[i][c][0])#= getAngle(cnt[i][j - 4][0],
                         #                                  cnt[i][j][0],
                         #                                  cnt[i][j + 4][0])
        cnt_splitted, mask2, boll = compare_angle(cnt, i, j, Angle, -90, -85, pretraitement_1_copy)
        if boll:
            if len(cnt_new) == 0:
                cnt_new = cnt_splitted
            else:
                cnt_new.extend(cnt_splitted)

            if Show_me:
                cv2.imshow("the region splited aprear with a black vertical line", mask2)
                cv2.waitKey(0)
    for t in range(1, 4):
        j = len(cnt[i]) - t
        Angle = getAngle(cnt[i][j][0],
                                                           cnt[i][j][0],
                                                           cnt[i][t - 1][0])
        cnt_splitted, mask2, boll = compare_angle(cnt, i, j, Angle, -90, -85, pretraitement_1_copy)
        if boll:
            if len(cnt_new) == 0:
                cnt_new = cnt_splitted
            else:
                cnt_new.extend(cnt_splitted)
    #dictionnary, correspondance = update_correspondance_face_candidate_structure(cnt_new, i, face_candidates,correspondance)
    return cnt_new#dictionnary, correspondance