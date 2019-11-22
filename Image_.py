import re
#import os

#import methodes_implementation
import cv2
# import the necessary packages
"""from imutils import paths
import argparse
import pandas as pd
import Pretraitement
#import shape_description_tools
import pprint
import time
#image en cours de detection juste après le
#calcule des vecteur de caractéristique
#est définit par cette strucutre
#
"""
import numpy as np
import os
class Image_:
    def __init__(self, jpg,bmp,imgjp, imgbp):
            self.name = jpg
            self.path=bmp
            self.image_jpg=imgjp
            self.image_bmp=imgbp
    def get_name(self):
        return self.name
    def get_path_bmp(self):
        path_ = self
        path_[0].split("/seg")[0].split("\\split")[0] + "img_" + path_[1].split('.jpg')[0] + '.bmp'
    def read_image_bmp(self):
        img_bmp = cv2.imread(self.path+"seg/"+self.name+".bmp",0)
        return img_bmp
    def read_image_jpg(self):
        img_jpg = cv2.imread(self.path+"big/"+self.name+".jpg")
        return img_jpg
    def show_image_bmp(self, img):
        cv2.imshow("name= "+self.name,img)
        cv2.waitKey()
    def show_image_bmp(self):
        cv2.imshow("name= "+self.name,self.image_bmp)
        cv2.waitKey(0)
    def show_image_jpg(self):
        cv2.imshow("name= " + self.name, self.image_jpg)
        cv2.waitKey(0)
    @staticmethod
    def show_img(path, name):
        cv2.imshow("name"+str(name),path)
        cv2.waitKey(0)
    @staticmethod
    def initializer( path):
        """print(os.path.isfile(re.sub("/big", "/seg", path).replace(".jpg",".bmp")))
        print(os.path.isfile(path))
        print((re.sub("/big", "/seg", path)))
        """
        if (os.path.isfile(path) and path.endswith(".jpg") and os.path.isfile(re.sub("/big", "/seg", path).replace(".jpg",                                                                            ".bmp"))):  # "\\big" in str(path) or "/big" in str(path) and str(path).endswith(".jpg")) and (os.path.isfile(re.sub("\\big", "\\seg", path)) or os.path.isfile(re.sub("/big", "/seg", path))):
            #print("jpg")
            path_ = str.split(str(path), "img_")
            name = str.split("img_" + path_[1], ".jpg")[0]
            path_ = path_[0].replace("big//", "").replace("big/", "")
            # path_=str.split(str(path_),"seg//")[0]
            path_bmp = path_ + "seg/"
            image_bmp = cv2.imread(path_bmp+name+".bmp", 0)
            image_jpg = cv2.imread(path)
            img = Image_(name, path_,
                        image_jpg, image_bmp)
            #Image.show_img(img.image_jpg, img.name)
            return img
        else:
            if (os.path.isfile(path) and path.endswith(".bmp") and os.path.isfile(re.sub("/seg", "/big", path).replace(".bmp",                                                                                ".jpg"))):  # "\\big" in str(path) or "/big" in str(path) and str(path).endswith(".jpg")) and (os.path.isfile(re.sub("\\big", "\\seg", path)) or os.path.isfile(re.sub("/big", "/seg", path))):
                #print("bmp")
                # name= "img_"+path.split("img_"," ")[1]
                # print(path.split("img_"," "))#.split("/big","")[0].split("\\big", "")[0]
                path_ = str.split(str(path), "img_")
                name=str.split("img_"+path_[1],".bmp")[0]
                path_=path_[0].replace("seg//","").replace("seg/","")
                #path_=str.split(str(path_),"seg//")[0]
                path_jpg = path_ + "big/"
                image_bmp = cv2.imread(path, 0)
                image_jpg = cv2.imread(path_jpg+name+".jpg")
                img = Image_(name, path_,
                             image_jpg,image_bmp)
                #gg = img.read_image_jpg()
                #Image_.show_img(gg,img.name)
                #print("ID=")
                #print(img.get_ID_img())
                return img
            else: return {}
    def get_image_bmp(self):
        return self.image_bmp
    def get_image_jpg(self):
        return self.image_jpg
    def get_path(self):
        return self.path

    def get_ID(self):
        ID = re.sub(r'.*FDDB_FDDB_S/', r'', self.path)

        ID = ID.replace("/", "").replace("\\", "")

        """i = str()
        for index in range(len(ID) - 1):
            i += ID[index]
        """
        return ID
    def get_ID_img(self):
        return self.get_ID(),self.name
    @staticmethod
    def from_ID_to_path(ID):
        f = list()
        for i in ID:
            f.append(i)
        f.insert(4,'/')
        f.insert(7,'/')
        f.insert(10,'/')
        g=str
        for i in f:
            g + g + str(i)
        path=g
        return path