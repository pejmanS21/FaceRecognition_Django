# Face Recognition

<br>

## [Module](./face_recognition/face_encoder/face_encoder.py)


```python
from face_encoder import FaceEncoder

# Load the module
FE = FaceEncoder('/path/to/main_folder_people/', img_size=160, recognition_threshold=0.3)

# Read images in people folder to create a database
encodes_db = FE.db_prepare(show_face=True, save_file=True, encoding_file_path='./data/encodes_db.pt')

"""--- Different ways to send an input to get results ---"""

# Image path as string
image = FE.recognizer('/path/to/image', encodes_db, pil_write=True)
# PIL.Image
image = FE.recognizer(Image.open('/path/to/image'), encodes_db, pil_write=True)
# numpy image
image = FE.recognizer(np.array(unknown_image), encodes_db, pil_write=True)
```
<br>

## Colab

>***For testing the module yourself, open the prepared jupyter notebook in colab via the following link*** 

<a href="https://colab.research.google.com/drive/1hQkR1-QIyMRHnsTM2NBc75UOgoF-cpRc?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<br>

## Usage 

>***First***: install requirements.txt

    $ pip3 install -r requirements.txt

then in your local computer to start `Django` server run following commands in your terminal:

1- **makemigrations**

    $ python3 manage.py makemigrations

2- **migrate**

    $ python3 manage.py migrate

>Optional

- **createsuperuser**

    $ python3 manage.py createsuperuser

3- **runserver**

    $ python manage.py runserver 0.0.0.0:8000

<br>

## Docker

    $ docker-compose up --build

<br>

## UI

![img](./media/FRDjango.gif)

## Reference

For the `FaceRecognition` module I use the following repo as my reference: 
https://github.com/miladlink/torch_face
