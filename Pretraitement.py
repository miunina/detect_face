import copy
#import time
import cv2

#var toujours présente / constantes
choix_Morphological ={0:"OPEN",1:"CLOSE",2:"CLOSE_OPEN", 3:"OPEN_CLOSE"}
#list_morf = {0:"MORPH_RECT",1:"MORPH_CROSS",2:"Morph_ELLIPSE"}
_list_structurant_element_ = {0:"RECT",1:"CROSS",2:"ELLIPSE"}
_list_choix_ = {2:("CLOSE",cv2.MORPH_CLOSE),3:("OPEN", cv2.MORPH_OPEN),4:("DILATE",cv2.MORPH_DILATE),5:("ERODE", cv2.MORPH_ERODE)}

#var choisit optionnelement
rows = 4
columns= 6


class pretraitement:

    def Morphological_eliminating_noise(img,Kernel, Structurant_Element,iteration, choix, Show_Me=False ):


        #eliminating noise on external region contours
        mask_1 = copy.deepcopy(img)
        if Show_Me:
                    cv2.imshow('=',mask_1)
                    cv2.waitKey(0)
        k = cv2.getStructuringElement(Structurant_Element, (Kernel,Kernel))
        mask_1 = cv2.morphologyEx(mask_1,choix, kernel=k, iterations=iteration)
        if Show_Me:
                cv2.imshow('=', mask_1)

        return mask_1
    
    def Threshold_Binarisation(img_Blured):
        cet, Thresh = cv2.threshold(img_Blured, 0, 255, cv2.THRESH_BINARY)
        return Thresh
                        #=======================================#
                         #================Lissage===============:
                           #=======================================#
    def lissage_gaussianBlur(img, Kernel = 7, sigma = 0, iteration = 11, Show_Me=False):
        gaussian_blur = copy.deepcopy(img)
        for i in range(1, iteration, 2):

            gaussian_blur = cv2.GaussianBlur(gaussian_blur, (Kernel, Kernel), sigma)
            if Show_Me:
                cv2.imshow('=',gaussian_blur)
                cv2.waitKey(0)
        return gaussian_blur

    def lissage_median(img, kernel = 7, iteration = 11, write_me=False, path="", name=""):
        median_blur = img.copy()
        median_blur_to_write = img.copy()
        if write_me:
            for i in range(1, iteration, 2):
                median_blur = cv2.medianBlur(median_blur, kernel)
                median_blur_to_write = cv2.medianBlur(median_blur,kernel)
                cv2.putText(median_blur_to_write,"median blur stape k = "+str(i),(int(img.shape[1]/2)),int(img.shape[0]/2),(255,120,0))

            cv2.imwrite(path, median_blur)
            cv2.waitKey(0)
        else:
         for i in range(1, iteration, 2):

            median_blur = cv2.medianBlur(median_blur, kernel)

        return median_blur
                     #=======================================#
                         #================Eliminiation du bruit===============:
                           #===================================Entrée : l'image binaire original et ses dimensions
                             #=======================================Sortie :nouvelle image et nouvelle image avec edges  #

def EliminateNoise(Img, Show_Me = False):
    img_B = Img.copy()
    _list_structurant_element_ = {0:"RECT",1:"CROSS",2:"ELLIPSE"}
    _list_choix_ = {2:("CLOSE",cv2.MORPH_CLOSE),3:("OPEN", cv2.MORPH_OPEN),1:("DILATE",cv2.MORPH_DILATE),0:("ERODE", cv2.MORPH_ERODE)}
    #erode :
    elt_structurant =cv2.MORPH_CROSS
    _Kernel = 3
    __choix = cv2.MORPH_ERODE
    iteration__ = 3
    mm,_ = pretraitement.Morphological_eliminating_noise(img_B,_Kernel ,elt_structurant,iteration__ , __choix)

    #open :
    elt_structurant = cv2.MORPH_CROSS
    _Kernel = 3
    __choix = cv2.MORPH_OPEN
    iteration__ = 3
    mm,_ = pretraitement.Morphological_eliminating_noise(mm,_Kernel ,elt_structurant,iteration__ , __choix)

    ##dilate :
    #_Kernel = 3
    #__choix = 4
    #iteration__ = 3
    #mm,ll = pretraitement.Morphological_eliminating_noise(mm,_Kernel ,elt_structurant,iteration__ , __choix)
    # close :
    elt_structurant = cv2.MORPH_CROSS
    _Kernel = 3
    __choix =cv2.MORPH_CLOSE
    iteration__ = 3
    mm, _ = pretraitement.Morphological_eliminating_noise(mm, _Kernel, elt_structurant, iteration__, __choix)

    #median :
    _Kernel = 3
    iteration__ =3
    mm = pretraitement.lissage_median(mm, _Kernel, iteration__)
    #gauss :
    _Kernel = 3
    iteration__ =3
    sigma = 0

    mm= pretraitement.lissage_gaussianBlur(mm, _Kernel, sigma, iteration__)
    #threshold :
    thresh = pretraitement. Threshold_Binarisation(mm)
    #cv2.imshow('thresh=',thresh)
    #cv2.imshow('=',thresh)

    #cv2.destroyallwindows()
    #cv2.waitKey(0)
    return thresh, mm

def EliminateNoise2(org1, Show_Me = False):
    #Show_Me = True
    iteration__ =11#9#11
    org0=org1.copy()
    img_B = org0
    _Kernel = 3
    sigma = 5


   

    _list_structurant_element_ = {0:"RECT",1:"CROSS",2:"ELLIPSE"}
    _list_choix_ = {2:("CLOSE",cv2.MORPH_CLOSE),3:("OPEN", cv2.MORPH_OPEN),1:("DILATE",cv2.MORPH_DILATE),0:("ERODE", cv2.MORPH_ERODE)}

    #close :
    _Kernel = 3
    iteration__ = 1#2
    elt_structurant = cv2.MORPH_ELLIPSE
    __choix = cv2.MORPH_CLOSE

    mm= pretraitement.Morphological_eliminating_noise(img_B,_Kernel ,elt_structurant,iteration__ , __choix)

    mm = img_B


    #erode :
    _Kernel = 3
    iteration__ = 1#3
    elt_structurant = cv2.MORPH_ELLIPSE
    __choix = cv2.MORPH_ERODE
    mm= pretraitement.Morphological_eliminating_noise(mm,_Kernel ,elt_structurant, iteration__,__choix)


    #open :

    _Kernel = 3
    iteration__ = 1#4#6
    elt_structurant = cv2.MORPH_ELLIPSE
    __choix = 3
    mm = pretraitement.Morphological_eliminating_noise(mm,_Kernel ,elt_structurant, iteration__,__choix)

    """# close :
    _Kernel = 3
    iteration__ = 3  # 2
    elt_structurant = cv2.MORPH_ELLIPSE
    __choix = cv2.MORPH_CLOSE
    """
    #mm = pretraitement.Morphological_eliminating_noise(mm, _Kernel, elt_structurant, iteration__, __choix)

    #median :
    _Kernel =3# 5
    iteration__ =5
    mm = pretraitement.lissage_median(mm, _Kernel, iteration__)

    #gauss
    _Kernel = 3
    iteration__ =5
    #5
    #sigma =# 5.8#5

    mm= pretraitement.lissage_gaussianBlur(mm, _Kernel, sigma, iteration__)
    #threshold :
    thresh = pretraitement. Threshold_Binarisation(mm.copy())

    #cv2.imshow('=', thresh)
    #cv2.imshow('=d',org0)

    # cv2.destroyallwindows()
    #cv2.waitKey(0)

    if Show_Me:
       #cv2.imshow('thresh=',thresh)
       cv2.imshow('=',thresh)

       #cv2.destroyallwindows()
       cv2.waitKey(0)
       cv2.destroyAllWindows()
    org = thresh
    #cv2.rectangle(org, (0, 0), (org.shape[1] , org.shape[0] ), 0, 3)
    return mm,org, org1


def EliminateNoise3(external_region, write_me=False):
    # pretreatment:
    external_region_copy = external_region.copy()
    # 1 = close open median gauss
    kernel = (7, 7)
    external_region = cv2.dilate(external_region, kernel, iterations=5)
    external_region = cv2.morphologyEx(external_region, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel), 5)

    kernel = (3, 3)
    iteration_ = 5
    ee = cv2.erode(external_region, kernel, iterations=2)

    # iteration_ = 1
    dilate = ee  # cv2.dilate(ee, kernel, iterations=iteration)
    kernel = 3
    # iteration_ = 3
    pretraitement_1 = pretraitement.lissage_median(dilate, kernel, 5)
    kernel = (3, 3)

    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel,3)
    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 7)
    high_thresh, pretraitement_1 = cv2.threshold(pretraitement_1, 126, 255,
                                                 cv2.THRESH_BINARY)
    return pretraitement_1
