import os
import cv2
import uuid
from os.path import exists


class Webcam :

    POS_PATH = os.path.join("data", "positive")
    NEG_PATH = os.path.join("data", "negative")
    ANC_PATH = os.path.join("data", "achor")

    def __init__(self, name):
        self.name = name
    
    def setupPath(self):
        paths = [self.POS_PATH, self.NEG_PATH, self.ANC_PATH]

        # Make the directories
        for path in paths:
            if (not exists(path)):
                os.makedirs(path)

    def buildWebcam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


        while True:
            success, image = self.cap.read()

            # Cut down frame to 250x250px
            image = image[120:120+250, 200:200+250, :]

            if success: 
                cv2.imshow("video", image)

            # Collect anchors
            if cv2.waitKey(1) & 0xFF == ord('a'):
                # Create the unique file path
                imgname = os.path.join(self.ANC_PATH, "{}.jpg".format(uuid.uuid1()))
                # Write out anchor image
                cv2.imwrite(imgname, image)

            # Collect positives
            if cv2.waitKey(1) & 0xFF == ord('p'):
                # Create the unique file path
                imgname = os.path.join(self.POS_PATH, "{}.jpg".format(uuid.uuid1()))
                # Write out positive image
                cv2.imwrite(imgname, image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the image
        self.cap.release()
        # Close the image
        cv2.destroyAllWindows()


    
    # def enableWebcam(self):
    #     os.system('Get-PnpDevice -FriendlyName *webcam* -Class Camera,image')
    #     os.system('Enable-PnpDevice -InstanceId (Get-PnpDevice -FriendlyName *webcam* -Class Camera -Status Error).InstanceId')
    
    # def disableWebcam(self):
    #     os.system('Disable-PnpDevice -InstanceId (Get-PnpDevice -FriendlyName *webcam* -Class Camera -Status OK).InstanceId')

    # def openWebcam(self):
    #     os.system('start microsoft.windows.camera:')

    # def camera():
    #     pass

  
w1 = Webcam("webcame1")

w1.buildWebcam()