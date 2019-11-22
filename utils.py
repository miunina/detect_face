
def findFolder():
    path="FDDB_FDDB_S_2\\"
    list_fichiers = list()
    for dossier, sous_dossiers, fichiers in walk(path):
        if dossier.endswith('seg'):
            for fichier in fichiers:
                list_fichiers.append(join(dossier, fichier))
    # print(list_fichiers)
    return list_fichiers



#l=findFolder()
def expand_build_image_dataset_1(path):
    image_ = creating_annotations.Image.initializer(path)


    if isinstance(image_, creating_annotations.Image):
        Img_jpg = image_.read_image_jpg()
        Img_bmp = image_.read_image_bmp()
        # convert the images to PIL format...
        Img_jpg = cv2.cvtColor(Img_jpg, cv2.COLOR_BGR2RGB)
        Img_bmp_mat = image_.get_image_bmp()
        Img_bmp_mat_ext = image_.get_image_bmp()
        #print(type(Img_bmp_mat))
        #print(Img_bmp_mat.dtype)

        n, m = image_.get_ID_img()
        test = glob.glob("**/" + n + m + "*", recursive=True)
        for t in test:
            os.remove(t)
        for j in range(5):
            if Img_bmp_mat.ndim == 2:
                Img_jpg2 = cv2.cvtColor(Img_bmp_mat.copy(), cv2.COLOR_GRAY2RGB)
            else:
                Img_jpg2 = Img_bmp_mat.copy()

                # print(cnt.longeurCnt,type(cnt_int))
                # print(isinstance(Img_methode1,creating_annotations.methode1))
                # print(cnt.get_vecteur_caracteristique())
                # print(cnt.anotated_labeled_image())#return dict_cnt, labeled pd

                # writing maskAll (Img_bmp_mat):
                # let s split the region of the maskAll (Img_bmp_mat) by creating methode1 object
            Img_methode1 = creating_annotations.methode1.initializer(Img_bmp_mat)
            # Img_jpg = Img_methode1.pretreatment()
            cnt, cnt_int = Img_methode1.create_contour()
            # Let s write the vector of characteristics:
            # cnt.save_vecteur_caracteristique("datas_csv/"+n + m +"_"+str(j)+"_5.csv")
            # Let s write external region internal region :
            mask2 = cnt.getMask()  # write_cnt_mask_by_mask_roi("temp_shape_properity/"+n + m + "_" + str(j) + "_0external_", Thickness=-1)
            external_region = cv2.cvtColor(cnt.getMask(), cv2.COLOR_RGB2GRAY)
            mask = cnt_int.getMask()  # write_cnt_mask_by_mask_roi("temp_shape_properity/"+n + m + "_" + str(j) + "_0internal_", Thickness=-1)

            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            if mask2.ndim == 3:
                mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2GRAY)

            # Let s split region of the mask external (Img_bmp_mat_ext) by creating methode1 object
            Img_methode1_ext = creating_annotations.methode1.initializer(Img_bmp_mat_ext)
            cnt_, cnt_int_ = Img_methode1_ext.create_contour()
            cnt_.write_cnt_mask_by_mask_roi("temps2/"+n+m+"_"+str(j)+"_",Color=(255,255,255),Thickness=-1)
            cnt_.save_vecteur_caracteristique("datas_csv/" + n + m + "_" + str(j) + "_6.csv", write_me=True,
                                              mask=Img_jpg2)

            # let s get the mask pretreated of maskAll
            # cnt_together = build_more_image_for_training.Contours(maskAll.copy(), maskAll.shape, cv2.RETR_EXTERNAL,#                                                    cv2.CHAIN_APPROX_SIMPLE)
            # cnt_together.write_cnt_Mask(n + m + "_" + str(j) + "_0together", Thickness=-1)
            # Let s get the mask pretreated of mask external region alone
            #cv2.imwrite("resultats/" + n + m + "_" + str(j) + "_0not_tog.bmp", external_region)

            external_region = Pretraitement.EliminateNoise3(external_region)
            # maskAll=cv2.cvtColor(maskAll,cv2.COLOR_RGB2GRAY)
            # Let s rebuild the mask external region from the splited contour for continuing amelioration
            # of the pretreatment changements :
            #cv2.imwrite("shape_description_draw/" + n + m + "_" + str(j) + "_7draw.jpg", Img_jpg2)
            external_region = cv2.bitwise_xor(external_region, mask)
            #cv2.imwrite("resultats/" + n + m + "_" + str(j) + "_0external_tog.bmp", external_region)
            Img_bmp_mat_ext = cv2.bitwise_not(external_region)

            Img_bmp_mat = Img_bmp_mat_ext

            # print(n,m)

            Img_jpg2 = Img_jpg.copy()

