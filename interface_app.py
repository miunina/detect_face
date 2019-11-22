
#from tkinter import *
from tkinter.filedialog import *
def main():

    import algorithme_principal_

    from PIL import Image
    from PIL import ImageTk
    import cv2
    import Image_
    global Frame1, Frame2, gray, img_
    global val, Seuillage
    global time_execution

    def clicRechercher():
        global Frame1, Frame2, gray, img_

        filepath = askopenfilename(title="Ouvrir une image")
        img_ = Image_.Image_.initializer(filepath)
        if len(filepath) > 0 and isinstance(img_, Image_.Image_):
            img = img_.get_image_jpg().copy()
            w = 200
            imgi = cv2.resize(img, (w, int((w / img.shape[1]) * img.shape[0])), 0, 0, cv2.INTER_NEAREST)
            img_RGB = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB)

            gray = img_.get_image_bmp().copy()
            gray_show=img_.get_image_bmp().copy()
            w = 200
            gray_show= cv2.resize(gray_show, (w, int((w / gray_show.shape[1]) * gray_show.shape[0])), 0, 0, cv2.INTER_NEAREST)

            # gray = img_.get_image_bmp()
            # img_RGB_copy = imgi.copy()

            # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Affichage image originale
            img_RGB = Image.fromarray(img_RGB)
            img_RGB = ImageTk.PhotoImage(img_RGB)
            Frame1.configure(image=img_RGB)
            Frame1.image = img_RGB

            # Affichage image niveau de gris
            graydisp = Image.fromarray(gray_show)
            graydisp = ImageTk.PhotoImage(graydisp)
            Frame2.configure(image=graydisp)
            Frame2.image = graydisp

    def clicTraiter():
        global val, Seuillage

        seuil = val.get()
        methode = Seuillage.get()

        #print((img_.get_path()))
        n = img_.get_ID()
        pathImgToWrite = "training_3/" + n
        thresh, time_execution = algorithme_principal_.regrouping([str(methode), n], img_.get_image_bmp().copy(),
                                                                  img_.get_image_jpg().copy())

        w = 200
        thresh = cv2.resize(thresh, (w, int((w / thresh.shape[1]) * thresh.shape[0])), 0, 0, cv2.INTER_NEAREST)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

        thresh = Image.fromarray(thresh)
        thresh = ImageTk.PhotoImage(thresh)
        Frame2.configure(image=thresh)
        Frame2.image = thresh
    fenetre = Tk()
    fenetre.geometry("800x700")
    #fenetre.configure(background="grey10")
    fenetre.title("face detection")
    fenetre.resizable(0, 0)
    background_image__ =Image.open("icons/fenetre05.png")
    background_image = ImageTk.PhotoImage(background_image__)
    background_label =Label(fenetre, image=background_image)

    #background_label.image = background_image
    background_label.place(x=0, y=0)#, relwidth=600, relheight=600)
    background_label.pack(side="bottom",expand="yes", fill="both")#Y)#(side=TOP, expand=N, fill=BOTH)

    #side = TOP, expand = Y, fill = BOTH
    #fenetre.wm_iconbitmap(False, '../icons/icons8-sort-left-32.png')
    #fenetre.iconbitmap(T

    # rue, '../icons/icons8-sort-left-32.png')
    P = PanedWindow(background_label, orient=HORIZONTAL)
    P.pack(side=TOP, expand=1, fill=Y)#, fill=BOTH

    Chargement = LabelFrame(P, text="Chargement d'une image", fg="gray99", bg="gray9")

    """
    photoimage = PhotoImage(file=r"../icons/browse-icon-20.png")
    photoimage=photoimage.subsample(3,3)
    boutonFile = Button(Chargement,text="Charger image",fg="gray99",bg="turquoise",image = photoimage,
                    compound = LEFT,width=140,command=clicRechercher)# text="Rechercher image", fg="light yellow", bg="gray40", command=clicRechercher)
    boutonFile.pack(pady=5, side=BOTTOM)
    
    """

    P.add(Chargement)

    """Reglage = LabelFrame(P, text="Réglage", fg="gray1", bg="gray99",height=50, width=50, padx=1, pady=1)
    PReglage = PanedWindow(Reglage, orient=VERTICAL,height=120, width=299)
    val = IntVar()
    # Scale(frameSeuil, variable=val, from_=0, to=255, orient=HORIZONTAL, tickinterval=50, length=200,
    #      label='Valeur du seuil').pack()
    # PReglage.add(frameSeuil)
    # barre = Frame(PReglage, height=100, width=2, bg="black")
    # barre.pack()
    # PReglage.add(barre)
    frameSeuilage = Frame(PReglage, width=5, height=5)
    frameSeuilage.configure(background="gray5")
    Label(frameSeuilage, fg="grey99", bg="gray1").pack()#text="Type de méthode",
    Seuillage = IntVar()
    R1 = Radiobutton(frameSeuilage, text="méthode arbre de décision",fg="gray20",bg="gray99",width=40, variable=Seuillage, value=0).pack(anchor=W)
    R2 = Radiobutton(frameSeuilage, text="méthode implémentée avec réseau de neuron",fg="gray20",bg="gray99",width=40, variable=Seuillage, value=1).pack(
        anchor=W)
    R3 = Radiobutton(frameSeuilage, text="cascades de haar", variable=Seuillage,fg="gray20",bg="gray99",width=40, value=2).pack(anchor=W)
    # R4 = Radiobutton(frameSeuilage, text="Seuil à 0", variable=Seuillage, value=3).pack(anchor=W)
    # R5 = Radiobutton(frameSeuilage, text="Seuil à 0 inversé", variable=Seuillage, value=4).pack(anchor=W)
    PReglage.add(frameSeuilage)
    PReglage.pack()
    photoimage_ = PhotoImage(file=r"../icons/face-detection(1).png")
    photoimage_ = photoimage_.subsample(3, 3)
    start_button = Button(Reglage, text="Traiter", fg="gray99",bg="turquoise",image=photoimage_,compound=LEFT,width=140, command=clicTraiter)#.pack(padx=5,pady=5, side=TOP)
    start_button.pack(padx=5,pady=5, side=BOTTOM)
    """

    PanelB = LabelFrame(P , fg="gray99", bg="gray5", width=500, height=500)

    photoimage = PhotoImage(file=r"icons/browse-icon-20.png")
    photoimage = photoimage.subsample(3, 3)
    boutonFile = Button(PanelB, text="Charger image",  fg="gray99", bg="gray9", image=photoimage,
                        compound=LEFT, width=140,
                        command=clicRechercher)  # text="Rechercher image", fg="light yellow", bg="gray40", command=clicRechercher)
    boutonFile.pack(pady=5, side=TOP)

    #Reglage = LabelFrame(P, text="Réglage", fg="gray1", bg="gray99",height=50, width=50, padx=1, pady=1)
    PReglage = PanedWindow(PanelB, orient=VERTICAL, height=120, width=299)

    val = IntVar()
    # Scale(frameSeuil, variable=val, from_=0, to=255, orient=HORIZONTAL, tickinterval=50, length=200,
    #      label='Valeur du seuil').pack()
    # PReglage.add(frameSeuil)
    # barre = Frame(PReglage, height=100, width=2, bg="black")
    # barre.pack()
    # PReglage.add(barre)
    frameSeuilage = Frame(PReglage, width=5, height=5)
    frameSeuilage.configure(background="gray99")
    Label(frameSeuilage, fg="grey1", bg="gray99").pack()  # text="Type de méthode",
    Seuillage = IntVar()
    R1 = Radiobutton(frameSeuilage, text="méthode arbre de décision", fg="gray99", bg="gray9", width=40,
                     variable=Seuillage, value=0).pack(anchor=W)
    R2 = Radiobutton(frameSeuilage, text="méthode implémentée avec réseau de neurone", fg="gray99", bg="gray9",
                     width=40, variable=Seuillage, value=1).pack(
        anchor=W)
    R3 = Radiobutton(frameSeuilage, text="viola and jones", variable=Seuillage, fg="gray99", bg="gray9", width=40,
                     value=2).pack(anchor=W)
    # R4 = Radiobutton(frameSeuilage, text="Seuil à 0", variable=Seuillage, value=3).pack(anchor=W)
    # R5 = Radiobutton(frameSeuilage, text="Seuil à 0 inversé", variable=Seuillage, value=4).pack(anchor=W)
    PReglage.add(frameSeuilage)
    PReglage.pack()

    Frame1 = Label(PanelB)

    Frame1.place(relx=0.45, rely=0.45, anchor=E)#.pack(padx=5, pady=5, side=LEFT, fill=X, expand=1)
    Frame2 = Label(PanelB)
    Frame2.place(relx=0.45, rely=0.45, anchor=W)#.pack(padx=5, pady=5, side=RIGHT, fill=X, expand=1)
    PanelB.place(relx=0.45, rely=0.45, anchor=CENTER)#ack(side=TOP, fill=X, expand=1)
    P.add(PanelB)

    photoimage_ = PhotoImage(file=r"icons/face-detection(1).png")
    photoimage_ = photoimage_.subsample(3, 3)#magenta4
    start_button = Button(PanelB, text="Traiter", fg="gray99", bg="gray9", image=photoimage_, compound=LEFT,
                          width=600, command=clicTraiter)  # .pack(padx=5,pady=5, side=TOP)
    start_button.pack(padx=100, pady=100, side=BOTTOM)
    #PReglage.pack()
    PanelB.pack()
    P.add(PanelB)


    # Frame3 = Label(text ="time execution",fg="gray1", bg="gray99", width = 12, height=1)
    # Frame3.pack(padx=5, pady=5, side=TOP)
    ar = ["Description de forme et arbre de decision", "réseau de neurone", "viola and jones"]
    # Frame4 = Label(fg="gray1", bg="gray99", width = 17, height=1)
    # Frame4.pack(padx=1, pady=1, side=TOP)
    # global time_execution
    # FrameTime = Label(LabelFrame(P, text=time_execution.get(), width=16,height=3))
    # FrameTime.pack(padx =2, pady=2, side=RIGHT )

    fenetre.mainloop()
if __name__ == '__main__':
    main()