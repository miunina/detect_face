import cv2
import numpy as np
import propiété_forme
import train
import Pretraitement
import re
import pickle
import methodes
import time
import imutils
import copy
import Image_
import os
import glob
dossier = {
    "0":"result/result0/","1":"result/result1/","2":"result/result2/"


}
class Face:
    def __init__(self, df=list(dict())):
        self.faces = df

    def add(self, k):
        # if len(self.faces)<=1:
        #print(k)
        self.faces.append(k)

    def draw_faces(self, img, i, color=(0, 0, 255)):
        #print("type=")
        #print(type(self.faces[i]))
        cnt = self.faces[i].get("contour")
        m = self.faces[i].get("methode")
        mm = np.zeros(img.shape,np.uint8)
        x, y, w, h = propiété_forme.propriété_forme(cnt).StraigthBoundingBox()
        if "2" in m:
            color = (0, 255, 0)
        if "3" in m:
            color = (255, 0, 0)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        return img
    def delet_attribute(self):

        i=len(self.faces)-1
        while i >= 0:

            #print(i)
            #print(len(self.faces))

            self.faces.pop(i)
            i=i-1# .clear())#self.faces.remove(


import pandas as pd
def is_face(o):
    f = o.get('label')
    return f == 1
import pandas as pd
import sys
import numpy as np
import cv2
def algorithme_haar(img, faces=Face(), ar="haarcascade_frontalface_alt.xml"):

    face_cascade = cv2.CascadeClassifier(ar)
    #eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    #print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces_ = face_cascade.detectMultiScale(gray, 1.3, 5)
    i=0

    for (x,y,w,h) in faces_:

        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        box_=((x,y),(x+w, y),(x+w,y+h), (x, y+h))
        box = np.array(box_, dtype="int")
        faces.add(dict(contour=box, methode="haar", indice=i))
        i+=1

        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #{    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return img, faces
#import Img_From_Foolder, methodes_implementation, stapes_requiered, shape_description_tools
import cv2#
import re#e
def algorithme_methode(cnt, cnt_int,pretraitement_1, shape, faces=Face(),methode="0", j=0, i=0):
    cnt_max = max(cnt, key=cv2.contourArea)
    cc=propiété_forme.propriété_forme(cnt_max).Area()
    for i in range(len(cnt)):

        v = propiété_forme.get_vecteur(cnt[i], shape[0], shape[1])

        ppd = pd.DataFrame(v, index=[0])

        #p = train.decision(ppd)

        if v.get('label') ==1:  # ppd.get('label') is 0:
            # print("is face")
            mm = np.zeros(pretraitement_1.shape, dtype=np.uint8)
            cv2.drawContours(mm,[cnt[i]], -1,255, -1)
            #cv2.imshow("mathode "+str(i),mm)
            #cv2.waitKey(0)
            faces.add(dict(contour=cnt[i], prediction=ppd, methode="0", indice=i))

            # i+=1
        else:

            # if "2" not in methode:

            if v.get("ratio") >= 1 and (v.get("Area")) >= (cc*75)/100 :#and v.get("Area") > 15:
                # new_cnt = methode_2(cnt[i],shape)
                #print("methode2")
                new_cnt = methodes.methode2(cnt, i,  pretraitement_1)
                if len(new_cnt)>0:
                 cnt_max_new = max(new_cnt, key=cv2.contourArea)
                 cc_new = propiété_forme.propriété_forme(cnt_max_new).Area()
                 for j in range(len(new_cnt)):
                    mask = np.zeros(pretraitement_1.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [new_cnt[j]], -1, 255, -1)

                    #cv2.imshow(str(j), mask)
                    v = propiété_forme.get_vecteur(new_cnt[j], shape[0], shape[1])

                    ppd = pd.DataFrame(v, index=[0])
                    #p = train.decision(ppd)
                    if v.get('label') == 1 and (v.get("Area")) >= (cc_new*40)/100:  # ppd.get('label') is 0:
                        #print("is face")
                        #m2 = np.zeros(pretraitement_1.shape, dtype=np.uint8)
                        #cv2.drawContours(mm2, [new_cnt[j]], -1, 255, -1)
                        #cv2.imshow("mathode 2" + str(j), mm2)
                        #cv2.waitKey(0)
                        faces.add(dict(contour=new_cnt[j], prediction=ppd, methode= "2", indice=j))
                #cv2.waitKey(0)
                # algorithme( new_cnt, shape,faces, methode + "2", i)
                # i+=1
                pass

            if v.get("ratio") < 1 and v.get("Area") > 15 :# and v.get("Area") > 15:

                cnt_adjacent = methodes.methode3(cnt, i,  pretraitement_1 ,cnt_int)
                for j in range(len(cnt_adjacent)):


                  if len(cnt_adjacent[j])<5:
                      pass
                  else:
                    mm3 = np.zeros(pretraitement_1.shape, dtype=np.uint8)
                    cv2.drawContours(mm3, [cnt_adjacent[j]], -1, 255, -1)
                    #cv2.imshow("mathode 3" + str(j), mm3)
                    #cv2.waitKey(0)

                    v = propiété_forme.get_vecteur(cnt_adjacent[j], shape[0], shape[1])


                    ppd = pd.DataFrame(v, index=[0])
                    p = v.get("label")
                    if p==1  and v.get('Area') > 9:#v.get('label') == 1:  # ppd.get('label') is 0:
                        faces.add(dict(contour=cnt_adjacent[j], prediction=ppd, methode="3", indice=j))


                # i+=1
                pass
            else:
                #nofaces.add(dict(contour=cnt[i], prediction=ppd, methode=methode, indice=i))

                pass
                # new_cnt = methode_3(cnt[i],shape)
                # algorithme( new_cnt, shape,faces, methode + "3", i)

    return faces#, nofaces

def algorithme(cnt, cnt_int,pretraitement_1, shape, faces=Face(),methode="0", j=0, i=0):
    for i in range(len(cnt)):
        v = propiété_forme.get_vecteur(cnt[i], shape[0], shape[1])
        ppd = pd.DataFrame(v, index=[0])
        # columns=['solidity', 'convexity',
        #                               'similarity_ellipse', 'circularity', 'roundness',
        #                               'ratio_rotated', 'ratio', 'extend', 'Area', 'label'], index=[0])
        # print(ppd.shape)
        # print(ppd.ndim)
        # print(ppd.get('label'))
        p = train.decision(ppd)
        #print(p[0])
        #print("resultas=")
        #print(p)
        # print(ppd.get_value(0,-1, True))
        #print(v.get("label"))
        #print(p[0]==0.0)
        if p[0]==1.0:
            #v.get('label') == 1:
            #  ppd.get('label') is 0:
            # print("is face")
            faces.add(dict(contour=cnt[i], prediction=ppd,explore=p, methode="0", indice=i))
            # i+=1
        else:

            # if "2" not in methode:

            if v.get("ratio") >= 1  and v.get('Area') > 15:
                # new_cnt = methode_2(cnt[i],shape)
                #print("methode2")
                new_cnt = methodes.methode2(cnt, i, pretraitement_1)

                for j in range(len(new_cnt)):
                    mask = np.zeros(pretraitement_1.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [new_cnt[j]], -1, 255, -1)
                    #cv2.imshow(str(j), mask)
                    v = propiété_forme.get_vecteur(new_cnt[j], shape[0], shape[1])
                    ppd = pd.DataFrame(v, index=[0])
                    p = train.decision(ppd)
                    if p[0]==1.0  and v.get('Area') >= 9:
                        #v.get('label') == 1:
                        #ppd.get('label') is 0:
                        #print("is face")
                        faces.add(dict(contour=new_cnt[j], prediction=ppd,explore=p, methode= "2", indice=j))
                #cv2.waitKey(0)
                # algorithme( new_cnt, shape,faces, methode + "2", i)
                # i+=1
                pass

            if v.get("ratio") < 1 and v.get("Area") > 15  :
                #print("methode3")
                cnt_adjacent = methodes.methode3(cnt, i,  pretraitement_1 ,cnt_int)
                for j in range(len(cnt_adjacent)):
                  mask = np.zeros(pretraitement_1.shape, dtype=np.uint8)
                  #cv2.drawContours(mask, [cnt_adjacent[j]], -1, 255, -1)
                  #cv2.imshow(str(j), mask)
                  if len(cnt_adjacent[j])<5:
                      pass
                  else:
                    v = propiété_forme.get_vecteur(cnt_adjacent[j], shape[0], shape[1])
                    #print("methode="+str(v.get("label")))
                    ppd = pd.DataFrame(v, index=[0])
                    p = train.decision(ppd)
                    if p[0]==1.0  and v.get('Area') > 9:#v.get('label') == 1:  # ppd.get('label') is 0:
                        #print("is face")
                        faces.add(dict(contour=cnt_adjacent[j], prediction=ppd, explore=p,methode="3", indice=j))

                #cv2.waitKey(0)
                # i+=1
                pass
            else:
                #nofaces.add(dict(contour=cnt[i], prediction=ppd,explore=p, methode=methode, indice=i))
                pass
                # new_cnt = methode_3(cnt[i],shape)
                # algorithme( new_cnt, shape,faces, methode + "3", i)

    return faces#, nofaces


def get_ID(path):
    ID = re.sub(r'.*FDDB_S/', r'', path)

    ID = ID.replace("/", "").replace("\\", "")

    return ID


def pretreatment(external_region):
    # pretreatment:

    external_region_copy = external_region.copy()
    # 1 = close open median gauss
    kernel = (7, 7)
    external_region = cv2.dilate(external_region, kernel, iterations=2)
    external_region = cv2.morphologyEx(external_region, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel), 5)

    kernel = (3, 3)
    iteration_ = 5
    ee = cv2.erode(external_region, kernel, iterations=5)
    kernel = (3, 3)
    iteration_ = 5
    ee = cv2.morphologyEx(ee, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel),
                                 iterations=5)

    kernel = (3, 3)
    iteration_ = 2
    # ee= cv2.morphologyEx(ee, cv2.MORPH_CLOSE,
    #                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel),
    #                                   iterations=3)
    kernel = (3, 3)
    iteration_ = 1
    dilate = ee  # cv2.dilate(ee, kernel, iterations=iteration)
    kernel = 3
    iteration_ = 3
    pretraitement_1 = Pretraitement.pretraitement.lissage_median(dilate, kernel, 3)
    kernel = (3, 3)

    # pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel,3)
    # pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    # pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    pretraitement_1 = cv2.GaussianBlur(pretraitement_1, kernel, 3)
    high_thresh, pretraitement_1 = cv2.threshold(pretraitement_1, 126, 255,
                                                 cv2.THRESH_BINARY)
    return pretraitement_1


def methode1_phase1(img_prete0,path="", write_me=False):
    # argv[2]+"\\faces\\"+str(argv[3])
    # pathImgToWrite="training_3/"

    img_prete = img_prete0.copy()
    img_inv, img_prete, external_region, internal_region = split_region(img_prete, path=path,write_me=write_me)
    # pretreatment applicate on external region:

    pretraitement_1 = pretreatment(external_region)  # , img_inv, Img_Colored_Readed, pathImgToWrite)
    pretraitement_1_copy_ = pretraitement_1.copy()
    if write_me:
        cv2.imwrite(path+"pretreatment.bmp",pretraitement_1)
    #cv2.imshow("pretr 1", pretraitement_1)
    #cv2.imshow("pret copy 1", pretraitement_1_copy_)
    #cv2.waitKey(0)
    # img_inv, img_prete, mask_extern, mask_internal = split_region(pretraitement_1)
    return img_inv, img_prete, pretraitement_1, internal_region


def get_cnt(mask_extern, arg=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(mask_extern, cv2.RETR_EXTERNAL, arg)[0]


def get_mask_i(cnt_i, shape, color=(255), thickness=-1):
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [cnt_i], -1, color, thickness)
    return mask


def split_region(img, path="",write_me=False):
    high_thresh, img_prete0 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    lowThresh = 0.5 * high_thresh
    # get the inverse of the binary image
    img_prete=img_prete0.copy()
    img_inv = cv2.bitwise_not(img_prete)
    img = img_inv.copy()
    # find external contours
    Contour0, hei = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    # Mask to draw the contours external on it
    Mask = img_prete.copy()
    for i in range(len(Contour0)):
        cv2.drawContours(Mask, [Contour0[i]], 0, (127), 1)
    # geting the contour external mask
    mask_contour_externe = cv2.bitwise_xor(Mask, img_prete)
    high_thresh, thresh_mask_contour_externe = cv2.threshold(mask_contour_externe, 126, 255, cv2.THRESH_BINARY)

    # to get the filled external region of the external contour extracted we redraw it with get mask
    cnt = cv2.findContours(thresh_mask_contour_externe.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)[0]
    mask_extern = np.zeros(img_prete.shape, dtype=np.uint8)

    for i in range(len(cnt)):
        cv2.drawContours(mask_extern, [cnt[i]], -1, 255, -1)
    # xor between inversed image and mask extern to get holes region

    #cv2.waitKey(0)
    mask_internal = cv2.bitwise_xor(img_inv, mask_extern)
    if write_me:
        cv2.imwrite(path+"split_region.bmp",np.hstack((img_inv,img_prete, mask_extern, mask_internal)))
    #cv2.imshow("split_region.bmp",np.hstack((img_inv,img_prete, mask_extern, mask_internal)))
    #cv2.waitKey(0)
    return img_inv, img_prete, mask_extern, mask_internal


def regrouping(argv, imgb_, img_colored_,list_faces=list(),list_no_faces=list(), write_me=False):
    faces = Face()
    #nofaces=Face()

    #print("faces) ) =")
    #print(faces.faces)

    methode_chosen={"0":"notre méthode","1":"notre méthode avec réseau de neurone","2":"avec cascade de haar"}
    t_deb=time.time()

    #img = cv2.imread(img_, 0)
    #img_colored = cv2.imread(img_colored_)

    img=imgb_.copy()
    img_colored=img_colored_.copy()

    img_inv, img_prete, mask_extern, mask_internal = methode1_phase1(img, path=argv[1]+argv[2], write_me=argv[3])
    cnt = get_cnt(mask_extern)

    #cv2.imshow("org coulored",img_colored_)
    #cv2.waitKey(0)
    #cnt_intern = get_cnt(mask_internal)
    #mask_extern = cv2.threshold(mask_extern,50,255,cv2.THRESH_BINARY)

    if argv[0]=="0":
        faces= algorithme_methode(cnt, mask_internal,mask_extern, img.shape, faces, argv[1]+argv[2], argv[3])
        list_faces.append(faces)
        #list_no_faces.append(nofaces)

    else:

        if argv[0]=="1":
            faces = algorithme(cnt, mask_internal,mask_extern, img.shape, faces, argv[1]+argv[2], argv[3])
            list_faces.append(faces)
            #list_no_faces.append(nofaces)

        else:

            if argv[0]=="2":
                img, faces = algorithme_haar(img_colored.copy(), faces, argv[1]+argv[2], argv[3])
                list_faces.append(faces)


    for i in range(len(faces.faces)):
        faces.draw_faces(img_colored, i)
    if write_me:
        cv2.imwrite(dossier[argv[0]]+argv[1]+"_"+argv[2]+"_"+argv[0]+".jpg", cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))
    t_fin=time.time()
    #print(str(t_fin-t_deb))

    faces.delet_attribute()
    #nofaces.delet_attribute()
    del faces
    #del nofaces
    return img_colored,str(t_fin-t_deb)

#regrouping(["1",""], "img_818.bmp", "img_818.jpg")
#regrouping(["2",""], "img_818.bmp", "img_818.jpg")
"""regrouping(["1",""],cv2.imread("img_818.bmp",0), copy.deepcopy(cv2.imread("img_818.jpg")))
regrouping(["2",""],cv2.imread("img_818.bmp",0),copy.deepcopy(cv2.imread("img_818.jpg")))

regrouping(["0",""],cv2.imread("img_818.bmp",0), copy.deepcopy(cv2.imread("img_818.jpg")))
"""
#regrouping("0","img_818.bmp", "img_818.jpg")





def build_train():
    path = "temp/"
    lis_file= glob.glob("*.jpg")
    for renamee in lis_file:
        pre, ext = os.path.splitext(renamee)
        os.rename(pre, pre +".bmp")

def from_labeled_to_labeled(vector = pd.DataFrame(columns=['solidity', 'convexity',
                                                        'similarity_ellipse', 'circularity', 'roundness',
                                                        'ratio_rotated', 'ratio', 'extend', 'Area', 'label'])):
    vector.replace(1,np.nan)
    vector.replace(0,np.nan)


#print(l)
def read(path, methode="0"):
    image_ = Image_.Image_.initializer(path)


    if isinstance(image_, Image_.Image_):
        Img_jpg = image_.read_image_jpg()
        Img_bmp = image_.read_image_bmp()
        # convert the images to PIL format...
        Img_jpg = cv2.cvtColor(Img_jpg, cv2.COLOR_BGR2RGB)
        Img_bmp_mat = image_.get_image_bmp()
        Img_bmp_mat_ext = image_.get_image_bmp()
        n, m = image_.get_ID_img()
        test = glob.glob("**/" + n + m + "*", recursive=True)
        for t in test:
            os.remove(t)
        Img_jpg2 = Img_jpg.copy()
        #ID=image_.get_ID()
        # méthode 2:
        return methode,n,m,Img_bmp,Img_jpg
        #cv2.imwrite("resultats/" + n + m + "_0not_tog.bmp", external_region)
        #cv2.imwrite("shape_description_draw/" + n + m + "_7draw.jpg", Img_jpg2)
def findFolder(path="FDDB_S/"):

    list_fichiers = list()
    for dossier, sous_dossiers, fichiers in os.walk(path):
        if dossier.endswith('seg'):
            for fichier in fichiers:
                list_fichiers.append(os.path.join(dossier, fichier))
    # print(list_fichiers)
    return list_fichiers

def main():
    m="0"
    path="FDDB_S/"
    l=findFolder(path)
    list_faces = list()
    list_no_faces = list()
    write_me=True
    #file=open("result/result1/execution_time_neural_network.txt",'w')
    #file = open("annotate.txt", 'w')
    indice = 1
    for i in l:
        #build_image_dataset(i.replace("\\", "/"))
        methodee, n, m,Img_bmp, Img_jpg=read(i.replace("\\","/"), methode="0")
        #cv2.imwrite("org_fddb_s/"+n+m+".jpg",cv2.cvtColor(Img_jpg,cv2.COLOR_BGR2RGB))
        #cv2.imwrite("bmp_fddb_s/" + n + m + ".bmp", Img_bmp)
        time=regrouping([methodee, "methode1/"+n, m+"_"+str(indice)+"_", write_me],Img_bmp,Img_jpg,list_faces,list_no_faces)[1]
        print(time)
        #file.write(time+"\n")
        #file.write(str(indice)+" "+str(n+m)+"  "+str(time) + "\n")
        indice += 1

        #pass
if __name__=="__main__":
    main()