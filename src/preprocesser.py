import cv2


def preprocess(image):
    """
        1. Normalize the image by dividing by 255
        2. Resize the image to 256x256
    """
    image = image/255
    image = cv2.resize(image, (256,256))
    return image
    
if __name__ == '__main__':
    image = cv2.imread('C:\\Users\\sam\\Downloads\\20240118201536_1.jpg')
    image = preprocess(image)
    print(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)

