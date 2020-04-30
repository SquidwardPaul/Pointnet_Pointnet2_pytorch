import cv2
from datetime import datetime
import time
import argparse
import sys
import configparser

from openni import openni2
from openni import _openni2 as c_api

width = 640
height = 480
fps = 30
mirroring = True
compression = False
lenght = 300 #5 minutes


def write_files(dev):

    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                   resolutionX=width,
                                                   resolutionY=height,
                                                   fps=fps))
    color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                   resolutionX=width,
                                                   resolutionY=height,
                                                   fps=fps))
    depth_stream.start()
    color_stream.start()
    dev.set_image_registration_mode(True)
    dev.set_depth_color_sync_enabled(True)

    depth_stream.set_mirroring_enabled(mirroring)
    color_stream.set_mirroring_enabled(mirroring)

    actual_date = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3]
    rec = openni2.Recorder((actual_date + ".oni").encode('utf-8'))
    rec.attach(depth_stream, compression)
    rec.attach(color_stream, compression)
    rec.start()
    print("Recording started.. press ctrl+C to stop or wait " + str(lenght) + " seconds..")
    start=time.time()
    try:
        while True:
            if (time.time()-start)>lenght:
                break
    except KeyboardInterrupt:
        pass
    rec.stop()
    depth_stream.close()
    color_stream.close()
    dev.close()
    rec.close()
def readSettings():
    global width,height,fps,mirroring,compression,lenght
    config = configparser.ConfigParser()
    config.read('settings.ini')
    width = int(config['camera']['width'])
    height = int(config['camera']['height'])
    fps = int(config['camera']['fps'])
    mirroring = config.getboolean('camera','mirroring')
    compression = config.getboolean('camera','compression')
    lenght = int(config['camera']['lenght'])

def main():

    readSettings()

    try:
        if sys.platform == "win32":
            libpath = "lib/Windows"
        else:
            libpath = "lib/Linux"
        openni2.initialize(libpath)
        print("Device initialized")
    except:
        print("Device not initialized")
        return
    try:
        dev = openni2.Device.open_any()
        write_files(dev)
    except:
        print("Unable to open the device")
    try:
        openni2.unload()
        print("Device unloaded")
    except:
        print("Device not unloaded")


if __name__ == '__main__':
    main()