from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

team1 = "Real Madrid"
team2 = "Atletico Madrid"
possessions = {}
side_time = {}

possessions[team1] = 260
possessions[team2] = 40
possessions["primer tercio"] = 40
possessions["segundo tercio"] = 160
possessions["tercer tercio"] = 220
side_time["left"] = 15
side_time["right"] = 150

# Rutas de las imágenes de los escudos
ruta_escudo_team1 = "imagenes/escudos/" + team1 + ".png"
ruta_escudo_team2 = "imagenes/escudos/" + team2 + ".png"

# Cargar las imágenes de los escudos
img_escudo_team1 = Image.open(ruta_escudo_team1)
img_escudo_team2 = Image.open(ruta_escudo_team2)
gt_img = cv2.imread('inference/black.jpg')
gt_h, gt_w, _ = gt_img.shape
bg_img = gt_img.copy()
bg_img = cv2.line(bg_img, (0, (gt_h //3) - 7), (gt_w, (gt_h //3) - 7), (0,0,255), 12)
bg_img = cv2.line(bg_img, (0, ((gt_h // 3) * 2) + 7), (gt_w, ((gt_h // 3) * 2) + 7), (0, 0, 255), 12)

# Crear los subplots
fig, axs = plt.subplots(4, 1, figsize=(15, 5))

# Subplot 1: Escudos de los equipos con porcentajes de posesión
axs[0].imshow(img_escudo_team1)
axs[0].text(1.1, 0.5, str(round(100 * (possessions[team1] / (possessions[team1] + possessions[team2])))) + "%", color='black', fontsize=24, ha='left', va='center',
                transform=axs[0].transAxes)
axs[0].axis('off')

axs[1].imshow(img_escudo_team2)
axs[1].text(1.1, 0.5, str(round(100 * (possessions[team2] / (possessions[team1] + possessions[team2])))) + "%", color='black', fontsize=24, ha='left', va='center',
                transform=axs[1].transAxes)
axs[1].axis('off')

# Subplot 2: Porcentaje de posesión en diferentes zonas del campo
axs[2].imshow(gt_img)

# Porcentaje de posesión en el lado izquierdo
left_percentage = str(round(100 * (side_time["left"] / (side_time["left"] + side_time["right"]))))
axs[2].text(0.28, 0.5, str(left_percentage) + "%", color='white', fontsize=16, ha='center', va='center',
                transform=axs[2].transAxes)

# Porcentaje de posesión en el lado derecho
right_percentage = str(round(100 * (side_time["right"] / (side_time["left"] + side_time["right"]))))
axs[2].text(0.72, 0.5, str(right_percentage) + "%", color='white', fontsize=16, ha='center', va='center',
                transform=axs[2].transAxes)

axs[2].axis('off')

# Subplot 3: Porcentaje de posesión en diferentes tercios del campo
axs[3].imshow(bg_img)

# Porcentaje de posesión en el lado izquierdo
upper_percentage = str(round(100 * (possessions["primer tercio"] / (possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"]))))
axs[3].text(0.5, 0.15, str(upper_percentage) + "%", color='white', fontsize=12, ha='center', va='center',
                transform=axs[3].transAxes)

# Porcentaje de posesión en el lado derecho
middle_percentage = str(round(100 * (possessions["segundo tercio"] / (possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"]))))
axs[3].text(0.5, 0.5, str(middle_percentage) + "%", color='white', fontsize=12, ha='center', va='center',
                transform=axs[3].transAxes)

lower_percentage = str(round(100 * (possessions["tercer tercio"] / (possessions["primer tercio"] + possessions["segundo tercio"] + possessions["tercer tercio"]))))
axs[3].text(0.5, 0.85, str(lower_percentage) + "%", color='white', fontsize=12, ha='center', va='center',
                transform=axs[3].transAxes)

axs[3].axis('off')

# Mostrar los subplots
plt.show()
#plt.savefig(clip_name[1:-4] + '_possessions.png', bbox_inches='tight')