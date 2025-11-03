import os
from PIL import Image
from network.colorization_model import ColorizationModel
from network.restoration_model import RestorationModel
from utils.exceptions import ModelNotLoadedException, ImageProcessingException

class ImageService:
    def __init__(self):
        try:
            self.colorizer = ColorizationModel()
        except Exception as e:
            self.colorizer = None
        try:
            checkpoint_path = "network/weights/damageRestoration/uformer_best.pth"
            if os.path.exists(checkpoint_path):
                self.restorer = RestorationModel(checkpoint_path)
            else:
                self.restorer = None
        except Exception as e:
            self.restorer = None

    def colorize(self, input_path: str, output_path: str):
        if not self.colorizer:
            raise ModelNotLoadedException()
        try:
            img = self.colorizer.colorize_with_eccv16(input_path)
            img.save(output_path)
            return output_path
        except Exception as e:
            raise ImageProcessingException(str(e))

    def restore(self, input_path: str, output_path: str):
        if not self.restorer:
            raise ModelNotLoadedException()
        try:
            img = self.restorer.process_image(input_path)
            img.save(output_path)
            return output_path
        except Exception as e:
            raise ImageProcessingException(str(e))
