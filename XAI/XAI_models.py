import torch
import numpy as np
from XAI.ImagePreparation import Image_Preparation
from XAI.BaseClass import get_model, BaseCAM
from XAI.cam_utils import get_2d_projection
from lime import lime_image


class LIME_Method(Image_Preparation):   #Blackbox Testing
    def __init__(self, model_path, image_path) -> None:
        super().__init__(image_path)
        self.model =  get_model(model_path)
        self.pil_transf = self.image_tensor()
        self.preprocess_transf = self.preprocess_transform()
        self.explainer = lime_image.LimeImageExplainer()

    def p_func(self,img):
        batch = torch.stack(tuple(self.preprocess_transf(i) for i in img), dim = 0)
        results = self.model.predict(batch, save = True, imgsz=640, verbose= False)
        conf = []
        for r in results:
            if len(r.boxes) > 0:
                conf.append([box.conf.item() for box in r.boxes])
            else:
                conf.append([0.0])    
        return np.array(conf)
    
    def start_explanation(self):
        explanation= self.explainer.explain_instance(
                np.array(self.pil_transf(self.image)), 
                self.p_func, top_labels=1, hide_color=0, num_samples=100)
        return explanation

    def __call__(self):
        return self.start_explanation()

    
    

class EIGEN_CAM(BaseCAM):    #Whitebox Testing
    def __init__(self, model, target_layers, task: str = 'od', #use_cuda=False,
                 reshape_transform=None):
        super(EIGEN_CAM, self).__init__(model,
                                       target_layers,
                                       task,
                                       #use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)
    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)

    

    

    


    

