import cv2

class Preprocesser:
    def __init__(self, image):
        self.image = image

    def preprocess(self):
        """
            1. Normalize the image by dividing by 255
            2. Resize the image to 256x256
        """
        self.image = self.image/255
        self.image = cv2.resize(self.image, (256,256))
        return self.image
    
if __name__ == '__main__':
    image = cv2.imread('C:\\Users\\sam\\Downloads\\20240118201536_1.jpg')
    preprocesser = Preprocesser(image)
    image = preprocesser.preprocess()
    print(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)

