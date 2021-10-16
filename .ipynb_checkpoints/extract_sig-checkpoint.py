
#https://github.com/ahmetozlu/signature_extractor

#Weitere Informationen zu KOMPONENTENANALYSE: https://homepages.inf.ed.ac.uk/rbf/HIPR2/label.htm
# Extrahiere Unterschriften von Bildern 

import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np


# Params Verwendung, um Pixel mit kleiner Größe zu löschen
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100

# Param Verw., um Pixel mit großer Größe zu löschen
constant_parameter_4 = 18


#Input Bild
img = cv2.imread('./Input/test.png',0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] #Binär sicherstellen


#Komponentenanalyse/Zusammenhangskomponente (Graphentheorie) (scikit-learn)
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img)

fig, ax = plt.subplots(figsize=(10,6))

'''
# plot the connected components (for debugging)
ax.imshow(image_label_overlay)
ax.set_axis_off()
plt.tight_layout()
plt.show()
'''

the_biggest_component = 0
total_area = 0
counter = 0
average = 0.0
for region in regionprops(blobs_labels):
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1
    # print region.area # (for debugging)
    # take regions with large enough areas
    if (region.area >= 250):
        if (region.area > the_biggest_component):
            the_biggest_component = region.area

average = (total_area/counter)
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))


a4_small_size_outliar_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))


a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))


pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)


component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (a4_big_size_outliar_constant)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0


plt.imsave('pre_version.png', pre_version)


# read the pre-version
img = cv2.imread('pre_version.png', 0)
# ensure binary
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# save the the result
cv2.imwrite("./Output/output.png", img)