import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import requests
from glob import glob
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt


def isnotebook():
    """check if code run in notebook
    Returns:
        bool: True= Jupyter notebook or Colab
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or 'Shell':
            return True   # Jupyter notebook or qtconsole or Colab
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if isnotebook():
    from tqdm.notebook import tqdm


class FaceEncoder(object):
    def __init__(self, main_folder_path: str = '../people/', img_size: int = 160,
                 recognition_threshold: float = 0.3) -> None:
        """
        FaceEncoder module
        >>> from face_encoder import FaceEncoder
        >>> FE = FaceEncoder('/path/to/main_folder_people/', img_size=160, recognition_threshold=0.3)

        :param main_folder_path: str default to '../people/'
        :param img_size: int default to 160
        :param recognition_threshold: float default to 0.3 (smaller value, higher accuracy)
        """

        self.main_folder_path = main_folder_path
        
        self.img_size = img_size
        self.recognition_threshold = recognition_threshold
        
        self.face_detector_model = MTCNN(image_size=self.img_size, margin=0.1)
        self.face_encoder_model = InceptionResnetV1(pretrained='vggface2').eval()

    def image_encoder(self, image, encodes_list: list = [], return_landmarks: bool = False,
                      show_face: bool = False) -> tuple or list:
        """

        :param image: np.ndarray, for encoding
        :param encodes_list: list, for append encoded features for same person, default is []
        :param return_landmarks: bool, return landmarks or not, used for visualization, default is False
        :param show_face: bool, show detected face during encoding, default is False
        :return: if return_landmarks is True, function will return encodes_list, landmarks as Tuple,
                else it'll return just encodes_list
        """
        image = np.array(image)
        landmarks = self.face_detector_model.detect(image)
        try:
            for i, box in enumerate(landmarks[0]):
                x1, y1, x2, y2 = list(map(lambda x: int(abs(x)), box))
                x1 = int(x1 / 1.2)
                y1 = int(y1 / 1.2)
                x2 = int(x2 * 1.8)
                y2 = int(y2 * 1.8)
                if (x2 - x1) * (y2 - y1) < ((224 ** 2) * 2):
                    pass
                # plt.imshow(image[y1:y2, x1:x2])
                # plt.show()
                face = image[y1:y2, x1:x2]
                try:    
                    face = self.face_detector_model(face)
                    if show_face:
                        plt.imshow(face.permute(1, 2, 0))
                        plt.show()
                    e = self.face_encoder_model(face.unsqueeze(0)).detach()
                    encodes_list.append(e)
                except AttributeError as e: 
                    print(e)
                    pass
            if return_landmarks:
                return encodes_list, landmarks

            return encodes_list
        except TypeError as e:
            pass

    def db_prepare(self, show_face: bool = False, save_file: bool = True,
                   encoding_file_path: str = './data/encoding.pt') -> dict:
        """
        >>> encodes_db = FE.db_prepare(show_face=True, save_file=True, encoding_file_path='./data/encodes_db.pt')
        :param show_face: bool, show detected face during encoding, default is False
        :param save_file: bool, save encodings database as .pt file, default is False
        :param encoding_file_path: str, where to save .pt file, default is './data/encodes_db.pt'
        :return: dict, encodings database, for future use.
        """

        encodings_db = dict()
        for person_name in tqdm(os.listdir(self.main_folder_path)):
            if os.path.isdir(os.path.join(self.main_folder_path, person_name)):
                if person_name == '.ipynb_checkpoints':
                    pass
            else: 
                pass
            encodes = []
            person_dir = os.path.join(self.main_folder_path, person_name)
            if '.DS_Store' in person_dir:
                pass
            name = person_dir.split('/')[-1]
            for img_name in glob(person_dir + '/*'):   # .jpg, .jpeg, .png
                # image = Image.open(img_name).convert('RGB')
                # image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
                image = read_rgb_image(image=img_name)
                encodes = self.image_encoder(image, encodes, show_face=show_face)
            
            if len(encodes) > 1:
                encoded_person = torch.mean(torch.vstack(tuple(i for i in encodes)), dim=0).unsqueeze(0)
            else:
                try:
                    encoded_person = encodes[0]
                except IndexError:
                    pass

            encodings_db[name] = encoded_person
        
        if save_file:
            torch.save(encodings_db, encoding_file_path)
        return encodings_db

    def recognizer(self, unknown_image, encoding_dict, pil_write: bool = False) -> np.ndarray:
        """
        >>> image = FE.recognizer('/path/to/image', encodes_db, pil_write=True)
        >>> image = FE.recognizer(Image.open('/path/to/image'), encodes_db, pil_write=True)
        >>> image = FE.recognizer(np.array(unknown_image), encodes_db, pil_write=True)

        :param unknown_image: str | PIL.Image.Image | np.ndarray | url, input image for prediction
        :param encoding_dict: dict | str, encoding database or path to .pt file
        :param pil_write: bool, write on images using PIL library default is False
        :return: np.ndarray, predicted image with writtings on,
        """
        
        if type(encoding_dict) == dict:
            encoding_db = encoding_dict
        elif type(encoding_dict) == str:
            encoding_db = torch.load(encoding_dict)
        
        unknown_image = read_rgb_image(unknown_image)
        unknown_encode, landmarks = self.image_encoder(unknown_image, [], show_face=False, return_landmarks=True)
        distances = dict()
        for i, box in enumerate(landmarks[0]):
            try:
                for key_name, known_encode in encoding_db.items():
                    dist = cosine(known_encode, unknown_encode[i])
                    distances[key_name] = dist
                min_score = min(distances.items(), key=lambda x: x[1])
                name = min(distances, key=lambda k: distances[k]) if min_score[1] < self.recognition_threshold else "Unknown"
                if name != 'Unknown':
                    name = name + '-{:.2f}'.format(1 - min_score[1])
                unknown_image = image_editor(unknown_image, name, box, pil_write=pil_write)
            except IndexError:
                # probably detect something as face
                pass
        # plt.imshow(unknown_image)
        # plt.show()
        return unknown_image


def image_editor(image: np.ndarray, text: str, box: np.ndarray, pil_write: bool = False) -> np.ndarray:
    """
    >>> image = image_editor(unknown_image, name, box, pil_write=True)
    :param image: np.ndarray, for draw box and writting on
    :param text: str, what to write on image
    :param box: list | np.ndarray, coordinate of box
    :param pil_write: bool, write on images using PIL library default is False
    :return: np.ndarray, edited image
    """

    x1, y1, x2, y2 = list(map(lambda x: int(abs(x)), box))
    scale = round(((x2 - x1) + 78) / 75)
    color = (244, 0, 20) if text == 'Unknown' else (49, 178, 97)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=5)
    if pil_write:
        image = cv2.rectangle(image, (x1, y2 + h + 2), (x1 + 500, y2), color, cv2.FILLED)
        image = Image.fromarray(image[...])
        draw = ImageDraw.Draw(image) 
        # specified font size
        font = ImageFont.truetype('./data/JetBrainsMono-Medium.ttf', 70) 
        # drawing text size
        draw.text((x1, y2 - 20), text, font=font) 
        image = np.array(image)
    else:
        image = cv2.rectangle(image, (x1, y2 + h + 2), (x1 + w, y2), color, cv2.FILLED)
        image = cv2.putText(image, text, (x1, y2 + h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 5)

    return image


def read_rgb_image(image) -> np.ndarray:
    """
    read image with RGB channels
    :param image: Any, image to open
    :return: np.ndarray, opened image
    """
    # PIL
    if isinstance(image, Image.Image):
        return np.array(image)

    # file or URL
    elif isinstance(image, str):
        image = Image.open(requests.get(image, stream=True).raw if str(image).startswith('http') else image).convert('RGB')
        return np.array(image)

    # OpenCV
    elif type(image) == np.ndarray:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        return image
    else:
        print('unknown image type')
        exit(-1)


def image_utils(image, show: bool = False, save: bool = False, save_path: str = './results.png') -> None:
    """

    :param image: Any, inputed image to show or save
    :param show: bool, show image via PIL.Image, default is False
    :param save: bool, save image via PIL.Image, default is False
    :param save_path: str, where to save image
    """
    # OpenCV
    if type(image) == np.ndarray:
        image = Image.fromarray(image[...])    # assume image is RGB, otherwise image[..., ::-1]
    # PIL
    elif isinstance(image, Image.Image):
        pass
    else:
        print('unknown image type')
        exit(-1)
    
    if show:
        image.show()
    
    if save:
        image.save(save_path)
