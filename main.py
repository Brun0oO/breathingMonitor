import cv2
import queue
import threading
import datetime
import matplotlib
import http.server
import socketserver
import math
from urllib.parse import urlparse, parse_qs
from io import BytesIO
import numpy as np
import PIL.Image as PImage
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DocRoot = "./"
ImgQueue = queue.Queue(3)
Ims = None


def calcMotion(region0, region1):
    """
    Calculates motion based on Lucas/Kanade optical flow method.
    Returns velocities in x,y
    """
    ra0 = np.diff(region0, axis=0)
    dyi = (ra0[:, :-1] + ra0[:, 1:])

    ra1 = np.diff(region0, axis=1)
    dxi = (ra1[:-1, :] + ra1[1:, :])

    diff = region1 - region0
    s1 = diff[:-1, :] + diff[1:, :]
    dt = s1[:, 1:] + s1[:, :-1:]
    x2 = (dxi * dxi).sum()
    xy = (dxi * dyi).sum()
    y2 = (dyi * dyi).sum()
    tx = (dxi * dt).sum()
    ty = (dyi * dt).sum()
    det = x2 * y2 - xy * xy
    det *= 0.5
    if abs(det) < 1E-3:
        return 0, 0
    u = (tx * y2 - ty * xy) / det
    v = (x2 * ty - xy * tx) / det
    mod = math.sqrt(u * u + v * v)
    if mod == 0:
        return 0, 0
    return u, v


def flatten(xs, ys):
    """
    Substracts linear fit from values ys and
    returns result and the value at the middle of the fitted values, ie the avg.
    """
    fitted = np.polyval(np.polyfit(xs, ys, 1), xs)
    return ys - fitted, fitted[len(xs) // 2]


def countCrossing(arr, avg):
    """
    Counts how many times the values cross the average.
    Only counts when going down.
    Returns the count
    """
    state = 0
    cnt = 0
    for i in arr:
        if state == 1 and i < avg:
            state = 2
            cnt += 1
        elif state == 2 and i > avg:
            state = 1
        elif state == 0:
            if i > avg:
                state = 1
            else:
                state = 2
    return cnt


"""
Use a global variable so we can do running average.
"""
BRate = 0


def genStripChart(figSize, xs, ys, flags):
    """
    Creates a strip char using matplotlib.
    First flattens the values and then counts the zero crossings.
    Returns the chart as PNG in binary form, which is then sent to the browser.
    """
    global BRate
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(111)
    xlen = len(xs)
    dxs = [0] * xlen
    for i in range(1, len(xs)):
        dxs[i] = (xs[i] - xs[0]).total_seconds()
    ys, avg = flatten(dxs, ys)
    cnt = countCrossing(ys, avg)

    if dxs[-1] > 0:
        cnt = cnt * 60.0 / dxs[-1]
    BRate = BRate + (cnt - BRate) * 0.3
    ax.plot(dxs, ys, flags)
    ax.set_ylim(-1.5, 1.5)
    ax.grid()
    ax.set_title("Breathing Rate = %.0f per min" % (BRate))
    ax.set_xlabel("seconds")
    plt.close()

    out = BytesIO()
    fig.savefig(out, format='png')
    out.seek(0)
    return out.read()


class ImageServerHandler(http.server.SimpleHTTPRequestHandler):
    """
    This class handles the HTTP request from clients.
    """

    def do_GET(self):
        global ImgQueue
        try:
            # ignore request to favicon
            if self.path.endswith('favicon.ico'):
                return
            parts = urlparse(self.path)
            qs = parse_qs(parts.query)
            req = parts.path[1:]
            reqHandler = self.handlerTable.get(req)
            if reqHandler != None:
                out, contype = reqHandler(self, req, qs)
            else:
                out, contype = self.serveFile(req, qs)
            self.send_response(200, "OK")
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.send_header("Content-Type", contype)
            self.end_headers()
            self.wfile.write(out)
        except Exception as e:
            try:
                self.send_response(503, "Error")
                self.end_headers()
                self.send_header("Content-Type", contype)
                self.wfile.write(bytes(out, "UTF-8"))
            except:
                print("Error", e)

    def getMimeType(self, fname):
        table = [['.js', 'text/javascript'],
                 ['.html', 'text/html'],
                 ['.css', 'text/css']]
        for suf, mtype in table:
            if fname.endswith(suf):
                return mtype
        return 'text/html'

    def serveFile(self, req, qs):
        global DocRoot
        req = req if len(req) > 0 else "/breathingMonitor.html"
        contype = self.getMimeType(req)
        with open(DocRoot + req, "r") as fh:
            content = fh.read()
        return bytes(content, "UTF-8"), contype

    def log_message(self, format, *args):
        return

    def getImage(self, req, qstr):
        """
        Gets image from the queue for display.
        Returns image data and mime type
        """
        im = ImgQueue.get()
        temp = BytesIO()
        im.save(temp, format="JPEG")
        return temp.getvalue(), "image/jpeg"

    def clicked(self, req, qstr):
        global Ims
        x = float(qstr['inpx'][0])
        y = float(qstr['inpy'][0])
        t = float(qstr['sigma'][0])
        t = min(1, max(0, t))
        Ims.startMeasurement(x, y, t)
        s = "start measurement at x %.0f y %.0f t %.2f" % (x, y, t)
        out = bytes(s, "UTF-8")
        contype = "text/plain"
        return out, contype

    def getChart(self, req, qstr):
        global Ims
        try:
            contype = "image/png"
            return genStripChart((5, 2.4), Ims.tData, Ims.mData, "r-"), contype
        except KeyboardInterrupt:
            return "Aborted", "text/plain"

    """
    Class table
    """
    handlerTable = {
        'get': getImage,
        'clicked': clicked,
        'chart': getChart
    }

    #
    # End of ImageServerHandler
    #


class ImageServer:
    """
    This class runs the camera thread and the HTTP request hanlder thread.
    """

    def __init__(self):
        """
        Initializes the camera connection
        """
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

        self.nlen = 256
        self.mData = np.array([0.0] * self.nlen)
        self.tData = [datetime.datetime.now()] * self.nlen
        self.measureX = 0
        self.measureY = 0
        self.first = True
        self.runAvg = 0
        self.runAvg1 = 0
        self.sigma = 0.3
        self.boxSize = 35
        self.imgWidth, self.imgHeight = 0, 0

    def camLoop(self):
        """
        Reads image from the camera stream and
        puts image in the queue.
        """
        global ImgQueue
        while True:
            ret, frame = self.cam.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PImage.fromarray(frame)
            try:
                if ImgQueue.full():
                    print("full")
                    ImgQueue.get()
                ImgQueue.put_nowait(img)
            except Exception:
                print("Queue error")

    def processingLoop(self):
        global ImgQueue
        while True:
            img = ImgQueue.get()
            w, h = img.size
            bwImg = np.array(img.convert("L").getdata()).reshape((h, w))
            self.digestImage(bwImg)

    def startMeasurement(self, x, y, t):
        size = self.boxSize
        x = min(max(0, x - size / 2), self.imgWidth - size - 1)
        y = min(max(0, y - size / 2), self.imgHeight - size - 1)
        self.measureX = int(x)
        self.measureY = int(y)
        self.sigma = t
        self.first = True

    def digestImage(self, imgBW):
        self.imgWidth, self.imgHeight = imgBW.shape
        if self.measureX == 0 or self.measureY == 0:
            self.measureX = int(self.imgWidth / 2)
            self.measureY = int(self.imgHeight / 2)
        size = self.boxSize
        x = self.measureX
        y = self.measureY
        region = imgBW[x:x + size, y:y + size]

        if self.first:
            self.first = False
            self.runAvg = 0
            self.runAvg1 = 0
            self.lastRegion = region
            self.mData = np.array([0.0] * self.nlen)

        u, col = calcMotion(region, self.lastRegion)
        self.lastRegion = region

        t1 = self.sigma
        t11 = 0.3
        self.runAvg1 = (col - self.runAvg1) * t1 + self.runAvg1
        col = self.runAvg = (self.runAvg1 - self.runAvg) * t11 + self.runAvg
        self.mData[0:-1] = self.mData[1:]
        self.tData[0:-1] = self.tData[1:]
        self.mData[-1] = col
        self.tData[-1] = datetime.datetime.now()

    def startCamera(self):
        thr = threading.Thread(target=self.camLoop)
        thr.daemon = True
        thr.start()

    def startProcessing(self):
        thr = threading.Thread(target=self.processingLoop)
        thr.daemon = True
        thr.start()

    def start(self, port):
        print("Open http://localhost:%d" % port)
        try:
            httpd = socketserver.TCPServer(("0.0.0.0", port), ImageServerHandler)
            try:
                httpd.serve_forever()
                httpd.shutdown()
            except KeyboardInterrupt:
                return
        except:
            print("Failed to start HTTP Server")
            return


#
# main
#
if __name__ == "__main__":
    Ims = ImageServer()
    Ims.startCamera()
    Ims.startProcessing()
    Ims.start(5010)

