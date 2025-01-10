from PIL import Image
from torchvision import transforms

class Image_Preparation:
    def __init__(self, path) -> None:
        self.image = self.get_image(path)

    def get_image(self, path):
        img = Image.open(path)
        return img.convert('RGB')
    
    def image_tensor(self):
        transf = transforms.Compose([
            transforms.Resize((640,640)),
            #transforms.ToTensor(),
        ])
        return transf
    
    def preprocess_transform(self):
        normalize = transforms.Normalize(mean=[0.485,0.456, 0.406],
                                         std=[0.229, 0.224,  0.225])
        transf = transforms.Compose([
         #   transforms.Resize((640,640)),
            transforms.ToTensor(),
            normalize
        ])
        return transf