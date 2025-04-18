# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements

from mmdet.apis import DetInferencer

logger = logging.getLogger(__name__)


class MmdetDetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "mmdet", "mmcv"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        
        # create model   
        self.inferencer = DetInferencer(model=self.config_path,
                                        weights=self.model_path, 
                                        device="cuda:0", 
                                        show_progress=False)
        model = self.inferencer.model

        self.CLASSES = model.dataset_meta['classes']

        # update model image size
        if self.image_size is not None:
            model.cfg.data.test.pipeline[1]["img_scale"] = (self.image_size, self.image_size)
        

        self.set_model(model)

    def set_model(self, model: Any):
        """
        Sets the underlying MMDetection model.
        Args:
            model: Any
                A MMDetection model
        """

        # set self.model
        self.model = model
        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, images: List, store_slice_results=False):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            images: List[np.ndarray, str, PIL.Image.Image]
                A numpy array that contains a list of images to be predicted. 3 channel image should be in RGB order.
        """

        if not isinstance(images, list):
            images = [images]

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        # perform inference
        for ind, image in enumerate(images):
            if isinstance(image, Image.Image):
                image = np.array(image)
            if isinstance(image, np.ndarray):
                # mmdet accepts in BGR order
                images[ind] = image[..., ::-1]
                # images[ind] = image

        # prediction_result = inference_detector(self.model, images)
        # print(f"running mmdet inferencer with batch size {len(images)}")
        prediction_result = self.inferencer(images, batch_size=len(images), )

        self._original_predictions = prediction_result['predictions']

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        if isinstance(self.CLASSES, str):
            num_categories = 1
        else:
            num_categories = len(self.CLASSES)
        return num_categories

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        if type(self.CLASSES) == str:
            # https://github.com/open-mmlab/mmdetection/pull/4973
            return (self.CLASSES,)
        else:
            return self.CLASSES

    def _create_object_predictions_from_original_predictions(
        self,
        offset_amounts: Optional[List[List[int]]] = [[0, 0]],
        full_shapes: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_predictions_per_image.
        Args:
            offset_amounts: list of list
                To remap the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[offset_x, offset_y],[offset_x, offset_y],...]
            full_shapes: list of list
                Size of the full image after remapping, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        category_mapping = self.category_mapping

        # parse boxes and masks from predictions
        # REWRITTEN FOR mmdetection v3 inferencer (which supports batching)
        num_categories = self.num_categories
        object_predictions_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            offset_amount = offset_amounts[image_ind]
            full_shape = None if full_shapes is None else full_shapes[image_ind]

            boxes = original_prediction["bboxes"]
            scores = original_prediction["scores"]
            labels = original_prediction["labels"]
            if self.has_mask:
                masks = original_prediction["masks"]

            object_predictions = []

            # process predictions
            n_detects = len(labels)
            # process predictions
            for i in range(n_detects):
                if self.has_mask:
                    mask = masks[i]

                bbox = boxes[i]
                score = scores[i]
                category_id = labels[i]
                category_name = category_mapping[str(category_id)]

                # ignore low scored predictions
                if score < self.confidence_threshold:
                    continue

                # parse prediction mask TODO: support
                # no parsing from RLE --> mmdet inferencer modified

                # if self.has_mask:
                #     if "counts" in mask:
                #         if can_decode_rle:
                #             bool_mask = mask_utils.decode(mask)
                #         else:
                #             raise ValueError(
                #                 "Can not decode rle mask. Please install pycocotools. ex: 'pip install pycocotools'"
                #             )
                #     else:
                #         bool_mask = mask

                #     # check if mask is valid
                #     # https://github.com/obss/sahi/discussions/696
                #     if get_bbox_from_bool_mask(bool_mask) is None:
                #         continue
                # else:
                #     bool_mask = None
                bool_mask = mask

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        bool_mask=bool_mask,
                        category_name=category_name,
                        offset_amount=offset_amount,
                        full_shape=full_shape,
                )
                object_predictions.append(object_prediction)
            object_predictions_per_image.append(object_predictions)
        # per image in batch
        self._object_predictions_per_image = object_predictions_per_image

            # for category_id in range(num_categories):
            #     category_boxes = boxes[category_id] # bleble! boxes list not indexed by catID --> thats mmdet v2 i believe
            #     if self.has_mask:
            #         category_masks = masks[category_id]
            #     num_category_predictions = len(category_boxes)

            #     for category_predictions_ind in range(num_category_predictions):
            #         bbox = category_boxes[category_predictions_ind][:4]
            #         score = category_boxes[category_predictions_ind][4]
            #         category_name = category_mapping[str(category_id)]

            #         # ignore low scored predictions
            #         if score < self.confidence_threshold:
            #             continue

            #         # parse prediction mask
            #         if self.has_mask:
            #             bool_mask = category_masks[category_predictions_ind]
            #             # check if mask is valid
            #             # https://github.com/obss/sahi/discussions/696
            #             if get_bbox_from_bool_mask(bool_mask) is None:
            #                 continue
            #         else:
            #             bool_mask = None

            #         # fix negative box coords
            #         bbox[0] = max(0, bbox[0])
            #         bbox[1] = max(0, bbox[1])
            #         bbox[2] = max(0, bbox[2])
            #         bbox[3] = max(0, bbox[3])

            #         # fix out of image box coords
            #         if full_shape is not None:
            #             bbox[0] = min(full_shape[1], bbox[0])
            #             bbox[1] = min(full_shape[0], bbox[1])
            #             bbox[2] = min(full_shape[1], bbox[2])
            #             bbox[3] = min(full_shape[0], bbox[3])

            #         # ignore invalid predictions
            #         if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
            #             logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
            #             continue

            #         object_prediction = ObjectPrediction(
            #             bbox=bbox,
            #             category_id=category_id,
            #             score=score,
            #             bool_mask=bool_mask,
            #             category_name=category_name,
            #             offset_amount=offset_amount,
            #             full_shape=full_shape,
            #         )
            #         object_predictions.append(object_prediction)
        #     object_predictions_per_image.append(object_predictions)
        # self._object_predictions_per_image = object_predictions_per_image
