from django.shortcuts import render
from django.http import HttpResponse
from .forms import PostForm
from .models import Post
import glob
import os
from .face_encoder import FaceEncoder, image_utils
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from termcolor import colored

import warnings
warnings.filterwarnings('ignore')


fe = FaceEncoder(main_folder_path='./face_recognition/data/people/', img_size=160, recognition_threshold=0.3)
if not os.path.exists('./face_recognition/data/encodings_db.pt'):
    print(colored("[WARNINIG] encodings_db doesn't exist.", 'yellow'))
    print(colored("[INFO] creating encodings database", 'blue'))
    encodings_db = fe.db_prepare(show_face=False, save_file=True,
                                encoding_file_path='./face_recognition/data/encodings_db.pt')
else:
    print(colored("[INFO] encodings database exist.", 'green'))
    file_exist = True


def main_page(request):
    return render(request, 'index.html')


def tutorial(request):
    return render(request, 'empty_display.html')


# Create your views here.
def image_upload(request):
    if request.method == 'POST':
        post_form = PostForm(data=request.POST, files=request.FILES)

        if post_form.is_valid():
            post_form.save()
            path = os.path.join('./media/images/', str(request.FILES['image']))
            try:
                image = fe.recognizer(path, encoding_dict='./face_recognition/data/encodings_db.pt', 
                                        pil_write=True)
                image_utils(image, save=True, save_path='./media/images/results.png')
            except TypeError:   # Can't detect any face in image
                pass

    else:
        post_form = PostForm()
    
    return render(request, 'upload.html', {'post_form': post_form})


def display(request):
    if request.method == 'GET':
        # image = Post.objects.all()
        try:
            list_of_files = glob.glob('./media/images/*')
            latest_file = max(list_of_files, key=os.path.getctime)
            latest_file = '../.' + latest_file  # for html visualization    '../../media/images/path/to/image'
            
            return render(request, 'display.html', {'image': latest_file})
        except ValueError as e:
            return render(request, 'empty_display.html')


def video_stream(request):
	return render(request, 'video.html')


def videoreader(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass


#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        (self.isTrue, self.frame) = self.capture.read()
        self.frame =  cv2.flip(self.frame, 1)
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        try:
            image = fe.recognizer(image, encoding_dict='./face_recognition/data/encodings_db.pt', 
                                    pil_write=True)
        except TypeError:
            pass
        _, image = cv2.imencode('.jpg', image)
        
        return image.tobytes()

    def update(self):
        while True:
            (self.isTrue, self.frame) = self.capture.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')