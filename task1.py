import cv2
import pytesseract

from matplotlib import pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

pic="3"
temp=f"temp{pic}"

img=cv2.imread(f"Task1/{pic}.jpg")


#
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def ocr(img):
    text=pytesseract.image_to_string(img)
    return text

#Preprocessing
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=10)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image, kernel, iterations=5)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def pic1_preprocessing(img_path):
    import cv2
    img=cv2.imread(img_path)
    cv2.imshow("original",img)
    cv2.waitKey(0)
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 206, 206, cv2.THRESH_BINARY)
    no_noise = noise_removal(im_bw)
    dilated_image = thick_font(no_noise)
    cv2.imshow("preprocessed", dilated_image)
    cv2.waitKey(0)
    t = ocr(dilated_image)
    print(t)

def pic2_preprocessing(img_path):
    import cv2
    img=cv2.imread(img_path)
    cv2.imshow("original",img)
    cv2.waitKey(0)
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 210, 206, cv2.THRESH_BINARY)
    no_noise = noise_removal(im_bw)
    dilated_image = thick_font(no_noise)
    cv2.imshow("preprocessed", dilated_image)
    cv2.waitKey(0)
    t = ocr(dilated_image)
    print(t)

def pic3_preprocessing(img_path):
    import cv2
    img=cv2.imread(img_path)
    cv2.imshow("original",img)
    cv2.waitKey(0)
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 216, 206, cv2.THRESH_BINARY)
    dilated_image = thick_font(im_bw)
    cv2.imshow("preprocessed", dilated_image)
    cv2.waitKey(0)
    t = ocr(dilated_image)
    print(t)


# pic1_preprocessing("Task1/1.jpg")

# pic2_preprocessing("Task1/2.jpg")

# pic3_preprocessing("Task1/3.jpg")

#
#
# gray_image = grayscale(img)
# cv2.imwrite(f"{temp}/gray.jpg", gray_image)
# thresh, im_bw = cv2.threshold(gray_image, 216, 206, cv2.THRESH_BINARY)
# cv2.imwrite(f"{temp}/bw_image.jpg", im_bw)
# no_noise = noise_removal(im_bw)
# cv2.imwrite(f"{temp}/no_noise.jpg", no_noise)
# eroded_image = thin_font(im_bw)
# cv2.imwrite(f"{temp}/eroded_image.jpg", eroded_image)
# dilated_image = thick_font(im_bw)
# cv2.imwrite(f"{temp}/dilated_image.jpg", dilated_image)
# #
# display(f"{temp}/gray.jpg")
# display(f"{temp}/bw_image.jpg")
# display(f"{temp}/no_noise.jpg")
# # display(f"{temp}/eroded_image.jpg")
# display(f"{temp}/dilated_image.jpg")
#
# t=ocr(f"{temp}/dilated_image.jpg")
# t=ocr(img)
# print(t)
#
# cv2.imshow("",img)
# cv2.waitKey(0)