
from tkinter import *
import algorithme_principal_
from tkinter.filedialog import *
from PIL import Image
from PIL import ImageTk
import cv2
import Image_
def open_window():
    fenetre = Tk()
    fenetre.geometry("620x600")
    fenetre.title("Seuillage d'image")
    P = PanedWindow(fenetre, orient=VERTICAL)
    P.pack(side=TOP, expand=Y, fill=BOTH)

    Chargement = LabelFrame(P, text="Chargement d'une image")
    boutonFile = Button(Chargement, text="Rechercher image", command=clicRechercher)
    boutonFile.pack(pady=5)
    P.add(Chargement)

    Reglage = LabelFrame(P, text="Reglage", padx=5, pady=5)
    PReglage = PanedWindow(Reglage, orient=HORIZONTAL)
    frameSeuil = Frame(PReglage)
    val = IntVar()
    # Scale(frameSeuil, variable=val, from_=0, to=255, orient=HORIZONTAL, tickinterval=50, length=200,
    #      label='Valeur du seuil').pack()
    # PReglage.add(frameSeuil)
    # barre = Frame(PReglage, height=100, width=2, bg="black")
    # barre.pack()
    # PReglage.add(barre)
    frameSeuilage = Frame(PReglage)
    Label(frameSeuilage, text="Type de methode").pack()
    Seuillage = IntVar()
    R1 = Radiobutton(frameSeuilage, text="methode arbre de decision", variable=Seuillage, value=0).pack(anchor=W)
    R2 = Radiobutton(frameSeuilage, text="methode implemente avec resau de neuron", variable=Seuillage, value=1).pack(
        anchor=W)
    R3 = Radiobutton(frameSeuilage, text="cascade de haar", variable=Seuillage, value=2).pack(anchor=W)
    # R4 = Radiobutton(frameSeuilage, text="Seuil à 0", variable=Seuillage, value=3).pack(anchor=W)
    # R5 = Radiobutton(frameSeuilage, text="Seuil à 0 inversé", variable=Seuillage, value=4).pack(anchor=W)
    PReglage.add(frameSeuilage)
    PReglage.pack()
    Button(Reglage, text="Traiter", command=clicTraiter).pack(padx=5, pady=5)
    Reglage.pack()
    P.add(Reglage)

    PanelB = LabelFrame(P, text="Affichage", width=500, height=120)
    Frame1 = Label(PanelB)
    Frame1.pack(padx=5, pady=5, side=LEFT)
    Frame2 = Label(PanelB)
    Frame2.pack(padx=5, pady=5, side=RIGHT)
    global time_execution
    FrameTime = Label(LabelFrame(P, text=time_execution.get(), width=16, height=3))
    FrameTime.pack(padx=2, pady=2, side=RIGHT)
    PanelB.pack()
    P.add(PanelB)
    fenetre.mainloop()


def clicRechercher():
    global Frame1, Frame2, gray, img_

    filepath = askopenfilename(title="Ouvrir une image")
    img_=Image_.Image_.initializer(filepath)
    if len(filepath) > 0 and isinstance(img_, Image_.Image_):
        img = img_.get_image_jpg()
        w = 300
        imgi = cv2.resize(img, (w, int((w / img.shape[1]) * img.shape[0])), 0, 0, cv2.INTER_NEAREST)
        img_RGB = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB)

        gray = img_.get_image_bmp()

        #gray = img_.get_image_bmp()
        #img_RGB_copy = imgi.copy()


        #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Affichage image originale
        img_RGB = Image.fromarray(img_RGB)
        img_RGB = ImageTk.PhotoImage(img_RGB)
        Frame1.configure(image=img_RGB)
        Frame1.image = img_RGB

        # Affichage image niveau de gris
        graydisp = Image.fromarray(gray)
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
    thresh, time_execution = algorithme_principal_.regrouping([str(methode), n],img_.get_image_bmp(), img_.get_image_jpg())

    w = 300
    thresh = cv2.resize(thresh, (w, int((w / thresh.shape[1]) * thresh.shape[0])), 0, 0, cv2.INTER_NEAREST)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)



    thresh = Image.fromarray(thresh)
    thresh = ImageTk.PhotoImage(thresh)
    Frame2.configure(image=thresh)
    Frame2.image = thresh

root =Tk()

root.resizable(0,0)
but1 = Button(CENTER, text="Start the detection process", command=open_window)
root.geometry("62x600+60*60")
root.title("Welcome to our detection application")