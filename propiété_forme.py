import math
import cv2
import Pretraitement
import imutils
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist
import pandas as pd
import math
import cv2

class propriété_forme:
    def __init__(self, cnt_i):
        self.region=cnt_i
        pass
    def Area(self):
        moment = cv2.moments(self.region)
        return round(float(moment["m00"]),4)
    def AreaPourcentage(self, shapei):
      
        x1_ = self.Area() * 100
        
        x2_ =shapei[0] * shapei[1]
        
        z_ = float(float(x1_)/float(x2_+ (10 ** -6) * 1))
 
        y_ =float(z_)
      
        x_=round(y_, 4)
        return x_
    def Perimeter(self):
        return round(float(cv2.arcLength(self.region,True)),4)
    def EllipseFitting(self):
        cnt = self.region
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        # orientation is the angle at which obj is directed. MA: major axis, ma: minor axis

        return (x, y), (MA, ma), round(float(angle), 4)

    def Center(self):
        M = cv2.moments(self.region)
        m= math.pow(10,-6)

        cX = int(M["m10"] / (M["m00"] + (10 ** -6) * 1))
        cY = int(M["m01"] / (M["m00"] + (10 ** -6) * 1))



        return cX, cY
    def DistanceFromCenter(self, withCoordinates=False):
        pass
    @staticmethod
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    def RotatedRatio(self):
        # compute the rotated bounding box of the contour===================================================
        box_rotated= cv2.minAreaRect(self.region)
        box_rotated = cv2.BoxPoints(box_rotated) if imutils.is_cv2() else cv2.boxPoints(box_rotated)
        box_rotated = np.array(box_rotated, dtype="int")
        b = box_rotated
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box_rotated = perspective.order_points(box_rotated)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box_rotated
        (tltrX, tltrY) = propriété_forme.midpoint(tl, tr)
        (blbrX, blbrY) = propriété_forme.midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = propriété_forme.midpoint(tl, bl)
        (trbrX, trbrY) = propriété_forme.midpoint(tr, br)
        points=((tltrX, tltrY),(blbrX, blbrY),(tlblX, tlblY),(trbrX, trbrY))
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        ratio_rotated=round(dA/(dB+0.0000000000001),4)
        return dA, dB, ratio_rotated, points
    def StraigthBoundingBox(self):
        x, y, w, h = cv2.boundingRect(self.region)
        return (round(float(x),4),round(float(y),4),round(float(w),4),round(float(h),4))
    def RotatedBoundingBox(self):
        cnt = self.region
        #rect = cv2.minAreaRect(cnt)


        box_rotated = cv2.minAreaRect(self.region)
        box_rotated = cv2.BoxPoints(box_rotated) if imutils.is_cv2() else cv2.boxPoints(box_rotated)
        box = np.array(box_rotated, dtype="int")
        #box = cv2.boxPoints(rect)  # data is float
        #box = np.int0(box)  # turn data into integer.


        return box,box

    def ExtremPoints(self):
        cnt = self.region
        # extreme points
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        return leftmost, rightmost, topmost, bottommost
    def SimilarityEllipse(self):
        cnt = self.region
        ellipse = cv2.fitEllipse(cnt)
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                int(ellipse[2]), 0, 360, 3)
        similarity = round(float(cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), cnt, 2, 0)), 4)
        return similarity
    def OrientationEllipse(self):
        a, b, c = self.EllipseFitting()
        return c


    def RatioStraigthRect(self):
        s_bounding_rect = self.StraigthBoundingBox()
        return round(float(s_bounding_rect[2]/s_bounding_rect[3]), 4)
    def ExtendStraigthRect(self):
         area = cv2.contourArea(self.region)
         x, y, w, h =self.StraigthBoundingBox()
         extent_ratioo=-1
         extent_ratioo = float(area) / (w * h)
         return round(float(extent_ratioo),4)
    def Orientation(self):
        moment = cv2.moments(self.region)
        return (180 / np.pi) * np.arctan2(2 * moment['mu11'], (moment['mu20'] - self.moment['mu02'])) / 2
    def Circularity(self):
        return round(float( (np.pi * 4 * self.Area()) / ((self.Perimeter() ** 2)+0.0000000000001)),4)
    def Eccentricity(self):
        moment = cv2.moments(self.region)
        a1 = (moment['mu20'] + moment['mu02']) / 2
        a2 = np.sqrt(4 * moment['mu11'] ** 2 + (moment['mu20'] - moment['mu02']) ** 2) / 2
        minor_axis = a1 - a2
        major_axis = a1 + a2
        eccentricity= np.sqrt(1 - minor_axis / major_axis)
        return round(float(eccentricity),4)
    def EccentricityAndAxis(self):
        moment = cv2.moments(self.region)
        a1 = (moment['mu20'] + moment['mu02']) / 2
        a2 = np.sqrt(4 * moment['mu11'] ** 2 + (moment['mu20'] - moment['mu02']) ** 2) / 2
        minor_axis = a1 - a2
        major_axis = a1 + a2
        eccentricity= np.sqrt(1 - minor_axis / major_axis)
        return round(float(eccentricity),4), minor_axis, major_axis


    #Contour descriptors structural

    def Solidity(self):

            return round(float(self.Area()/self.HullInfo()[2]),4)


    def HullInfo(self):
        cnt = self.region
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area = self.Area()
        hull_perimeter = cv2.arcLength(hull, False)
        perimeter = self.Perimeter()
        concavity = hull_perimeter/(perimeter+0.0000000000001)
        solidity_ratio = float(area) / (hull_area+0.0000000000001)
        #Also known as convexity. The proportion of the pixels in the convex hull that are also in the object. Computed as Area/ConvexArea
        hull_info1 = (round(float(solidity_ratio), 4),round(float(concavity), 4), round(float(hull_area), 4),
                      round(float(hull_perimeter), 4), hull)
        return hull_info1


    #=========================
    @staticmethod
    def _get_eucledian_distance(a, b):
        distance = math.sqrt( (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        return distance
    @staticmethod
    def calculate_euclidian_distance(m1, m2):

        x2 = m2[0][0]
        y2 = m2[0][1]
        m = tuple()
        m = m1
        y = y2-m[1]
        x = x2-m[0]
        z = math.sqrt(math.pow(x,2)+math.pow(y,2))
        return round(float(z), 4)
    
def get_vecteur_decision_tree_manual(self_, n, m, write_me=False, mask=[], Color=(205,50,0)):
        region_i = self_

        # calculate hull area and area
        # get the solidity
        # calculate hull perimeter and perimeter
        # get the concavity
        # get ratio
        # some area  calculation
        # this will stock some point of interest for labeling the region of image studied
        dfObj = dict()
        # calcutating characteristics shape:
        # get an object shape to calculate the properity needed
        shapee = propriété_forme(region_i)
        if len(shapee.region) <= 5:
            mask_ = np.zeros((n, m), np.uint8)
            cv2.drawContours(mask_, [region_i], -1, 255, -1)
            #cv2.dilate(mask_, (3, 3))

            region_i, hei = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # find the biggest area
            region_i = max(region_i, key=cv2.contourArea)
            shapee = propriété_forme(region_i)
        pourcentage_area = shapee.AreaPourcentage((n, m))
        Area = shapee.Area()


        hull_info = shapee.HullInfo()
        solidity = hull_info[0]
        circularity = shapee.Circularity()
        concavity = hull_info[1]
        extend = shapee.ExtendStraigthRect()
        ratio = shapee.RatioStraigthRect()
        dA, dB, ratio_rotated, points = shapee.RotatedRatio()
        if dA > dB:
            Major_Axis = dA
            bool = True
        else:
            Major_Axis = dB
            bool = False

        shape_returned = shapee.SimilarityEllipse()
        C_shape_returned = shape_returned <= 0.44
        roundness = round(4 * shapee.Area() / ((np.pi * Major_Axis ** 2)+0.0000000000001), 4)
        # ===========condition to verify :==================
        #rotated, elipse = shape.RotatedBoundingBox()
        pourcentage_area = shapee.AreaPourcentage((n, m))
        C1_Area = pourcentage_area >= 0.5
        C2_Area = pourcentage_area <= 90#70
        C3_Area = pourcentage_area >= 2
        C_circularity = circularity * -1 > 0.6
        C_concavity = concavity >= 0.67 and solidity >= 0.7
        C_extend = extend >= 0.5  # 0.43 and extend <= 0.77#0.5
        C_ratio = ratio > 0.85 and ratio <= 1.9

        C_solidity = solidity >= 0.8
        C_solidity_1 = solidity >= 0.8 and concavity >= 0.6
        C_ratio_rotated = ratio_rotated > 0.75 and ratio_rotated < 2.8  # >0.85
        label = 0

     
        x, y, w, h, = shapee.StraigthBoundingBox()
        if C1_Area:
            if C3_Area and C2_Area:
              if ratio > 0.47 and ratio < 1:
                #cv2.drawContours(mask, [self.cnt_i], -1, (255,
                # 255, 255), -1)
                if (C_ratio_rotated and C_solidity_1 and Area < 3 and C_circularity):
                        if write_me:
                            cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (25, 0, 199), 2)
                        #cv2.drawContours(mask,[self.cnt_i],-1,(205,99,0),1)
                        label = 1
                else:
                        if ((C_ratio_rotated) and (C_solidity or C_concavity) and Area >= 3):
                            if C_circularity or C_shape_returned:
                                label = 1
                                if write_me:
                                    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (25, 0, 199), 2)

                                #cv2.drawContours(mask, [self.cnt_i], -1, (205, 99, 0), 1)
                                label = 1
                            else:
                                if C_extend:
                                    label = 1
                                    if write_me:
                                        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (25, 0, 199), 2)

                                    #cv2.drawContours(mask, [self.cnt_i], -1, (205, 99, 0), 1)
                                    label = 1
                            #dfObj = np.array(
                            #1    [solidity, concavity,shape_returned,circularity, roundness, ratio_rotated, ratio, extend, Area, label])


        dfObj = dict(solidity=solidity, convexity=concavity,
                    similarity_ellipse=shape_returned,
                     circularity=circularity, roundness=roundness,
                 ratio_rotated=ratio_rotated, ratio=ratio, extend=extend, Area=Area, label=label)

        """dfObj = np.array([solidity, concavity,-1, circularity, roundness, ratio_rotated, ratio, extend, Area, label], dtype=np.float64)
        dfObj[2]=np.NaN
        """
        return dfObj

   

def get_vecteur(self, n, m, write_me=False, mask=[], Color=(205,50,0)):
        region_i = self

        # calculate hull area and area
        # get the solidity
        # calculate hull perimeter and perimeter
        # get the concavity
        # get ratio
        # some area  calculation
        # this will stock some point of interest for labeling the region of image studied
        dfObj = dict()
        # calcutating characteristics shape:
        # get an object shape to calculate the properity needed
        shapee =propriété_forme(region_i)
        if len(shapee.region) <= 5:
            mask_ = np.zeros((n, m), np.uint8)
            cv2.drawContours(mask_, [region_i], -1, 255, -1)
            mask_=cv2.dilate(mask_, (7, 7))
            mask_ = cv2.dilate(mask_, (7, 7))
            mask_ = cv2.dilate(mask_, (7, 7))


            region_i, hei = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # find the biggest area
            region_i = max(region_i, key=cv2.contourArea)
            shapee = propriété_forme(region_i)

        pourcentage_area = shapee.AreaPourcentage((n, m))
        Area = shapee.Area()


        hull_info = shapee.HullInfo()
        solidity = hull_info[0]
        circularity = shapee.Circularity()
        concavity = hull_info[1]
        extend = shapee.ExtendStraigthRect()
        ratio = shapee.RatioStraigthRect()
        dA, dB, ratio_rotated, points = shapee.RotatedRatio()
        if dA > dB:
            Major_Axis = dA
            bool = True
        else:
            Major_Axis = dB
            bool = False
        roundness = round(4 * shapee.Area() / ((np.pi * Major_Axis ** 2)+0.0000000000001), 4)
        # ===========condition to verify :==================
        #rotated, elipse = shape.RotatedBoundingBox()
        shape_returned = shapee.SimilarityEllipse()
        C_shape_returned = shape_returned <= 0.44
        pourcentage_area = shapee.AreaPourcentage((n, m))
        C1_Area = pourcentage_area >= 0.5
        C2_Area = pourcentage_area <= 90#70
        C3_Area = pourcentage_area >= 2
        C_circularity = circularity * -1 > 0.6
        C_concavity = concavity >= 0.67 and solidity >= 0.7
        C_extend = extend >= 0.5  # 0.43 and extend <= 0.77#0.5
        C_ratio = ratio > 0.85 and ratio <= 1.9
        C_solidity = solidity >= 0.8
        C_solidity_1 = solidity >= 0.8 and concavity >= 0.6
        C_ratio_rotated = ratio_rotated > 0.75 and ratio_rotated < 2.8  # >0.85
        label = 0

     
        x, y, w, h, = shapee.StraigthBoundingBox()
        if C1_Area:
            if C3_Area and C2_Area:
              if ratio > 0.47 and ratio < 1:
                #cv2.drawContours(mask, [self.cnt_i], -1, (255,
                # 255, 255), -1)
                if (C_ratio_rotated and C_solidity_1 and Area < 3 and C_circularity):
                        if write_me:

                            cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (25, 0, 199), 2)
                        #cv2.drawContours(mask,[self.cnt_i],-1,(205,99,0),1)
                        label = 1
                else:
                        if ((C_ratio_rotated) and (C_solidity or C_concavity) and Area >= 3):
                            if C_circularity or C_shape_returned:

                                if write_me:

                                    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (25, 0, 199), 2)

                                #cv2.drawContours(mask, [self.cnt_i], -1, (205, 99, 0), 1)
                                label = 1
                            else:
                                if C_extend:

                                    if write_me:

                                        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (25, 0, 199), 2)

                                    #cv2.drawContours(mask, [self.cnt_i], -1, (205, 99, 0), 1)
                                    label = 1


        dfObj = dict(solidity=solidity, convexity=concavity,
                    similarity_ellipse=shape_returned,
                     circularity=circularity, roundness=roundness,
                     ratio_rotated=ratio_rotated, ratio=ratio, extend=extend, Area=Area, label=label)

        """dfObj = np.array([solidity, concavity,-1, circularity, roundness, ratio_rotated, ratio, extend, Area, label], dtype=np.float64)
        dfObj[2]=np.NaN
        """

        return dfObj

   


