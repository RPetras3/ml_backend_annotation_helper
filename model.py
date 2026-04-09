from pyexpat import model

import os
import json
import pandas as pd
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

class annotation_image_label_types:
    """Helper class to define the label types for image annotation.
    These are defined as pandas DataFrame for easy manipulation easy to change into JSON format.
    """
    IMAGE_LABEL_TYPES = ['rectanglelabels', 'ellipselabels', 'polygonlabels', 'keypointlabels']
    IMAGE_LABEL_VALUES = [
        {IMAGE_LABEL_TYPES[0]: {'value': {
            'x': 0.0,
            'y': 0.0,
            'width': 0.0,
            'height': 0.0,
            'rotation': 0.0,
            IMAGE_LABEL_TYPES[0]: ['default']
        }}},
        {IMAGE_LABEL_TYPES[1]: {'value': {
            'x': 0.0,
            'y': 0.0,
            'width': 0.0,
            'height': 0.0,
            'rotation': 0.0,
            IMAGE_LABEL_TYPES[1]: ['default']
        }}},
        {IMAGE_LABEL_TYPES[2]: {'value': {
            'points': [],
            IMAGE_LABEL_TYPES[2]: ['default']
        }}},
        {IMAGE_LABEL_TYPES[3]: {'value': {
            'x': 0.0,
            'y': 0.0,
            IMAGE_LABEL_TYPES[3]: ['default']
        }}}
    ]
        
    def __init__(self, label_type: str):
        if label_type not in self.IMAGE_LABEL_TYPES:
            raise ValueError(f'Invalid label type: {label_type}. Supported types are: {self.IMAGE_LABEL_TYPES}')
        self.label_type = label_type
        self.from_name = 'label'
        self.to_name = 'image'
        self.type = label_type
        for label in self.IMAGE_LABEL_TYPES:
            if label_type == label:
                self.value = self.IMAGE_LABEL_VALUES[self.IMAGE_LABEL_TYPES.index(label)][label]['value']
                break
    
    



class ImageAnnotationApplier(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize any model or configuration here
        # If you have specific annotation rules, load them here
        self.annotation_rule = kwargs.get('annotation_rule', {})
        self.model_weights_file = 'model/deer_weights.pt'  # Placeholder for model weights file
        self.model_base = 'yolo5'  # Placeholder for the base model (e.g., YOLOv5)
        
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")
        
    def _apply_annotation_logic(self, task, **kwargs):
        """
        Internal method to apply the annotation to the image task.
        If 'annotation' is provided as an argument, use it.
        Otherwise, generate a default or model-based annotation.
        """
        annotations = kwargs.get('annotation', [])
        types = kwargs.get('type', [])
        
        ret_list = []
        for annotation, type in zip(annotations, types):
            if type in annotation_image_label_types.IMAGE_LABEL_TYPES:
                label_type = annotation_image_label_types(type)
                label_type.value.update(annotation)
                ret_list.append({
                    'from_name': label_type.from_name,
                    'to_name': label_type.to_name,
                    'type': label_type.type,
                    'value': label_type.value
                })
        
        # Fallback: Generate a default annotation (e.g., a placeholder rectangle)
        # This logic depends on your specific annotation type (rectangle, choices, etc.)
        return ret_list

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        if self.annotation_rule == 'train_first':
            # Extract the training tasks and their annotations from the context
            training_tasks = context.get('training_tasks', [])
            for task in training_tasks:
                # Copy data from the task off to the annotation_training_grounds directory for training
                name = os.path.basename(task['data']['image_url'])
                dest_path = os.path.join('annotation_training_grounds', 'train', 'images', name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                # The URL provided should be a local path to the file, so we can copy it directly
                os.system(f'cp {task["data"]["image_url"]} {dest_path}')
                print(f'Copied {task["data"]["image_url"]} to {dest_path}')
                # Copy the rest of the task data (e.g., annotations) to a corresponding JSON file for training
                annotation_dest_path = os.path.join('annotation_training_grounds', 'train', 'annotations', f'{os.path.splitext(name)[0]}.json')
                # Create a JSON entry for the annotation that matches YOLO/COCO format.
                os.makedirs(os.path.dirname(annotation_dest_path), exist_ok=True)
                with open(annotation_dest_path, 'w') as f:
                    json.dump(task['annotations'], f)
                print(f'Created annotation JSON at {annotation_dest_path}')

        
        for task in tasks:
            task['predictions'] = self._apply_annotation_logic(task, **kwargs)
        
        
        return ModelResponse(predictions=[])
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

