from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
import time

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
import tensorrt as trt
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger()



STD = np.array([
            0.5857040426,
            0.5523895348,
            0.5436823836],
        dtype=np.float32)

MEAN = np.array([
            0.9628600138,
            0.8621341349,
            0.7642968562],
        dtype=np.float32)
def preprocess(image, mean=MEAN, std=STD):
    # Mean normalization
    # mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    # stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / std

    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)


def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img

def get_engine_bindings(engine) -> dict:
    binding_shapes = {}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        print(f"Binding {i}: Name={name}, Shape={shape}, Input={is_input}")
        binding_shapes[name] = shape

    return binding_shapes

def load_engine_onnx(onnx_p="mmdeploy/work_dirs/trt_maskrcnn/end2end.onnx", 
                     engine_p="mmdeploy/work_dirs/trt_maskrcnn/from_onnx.engine"):
    ''' 
        builds and saves TRT engine to engin_path, based on ONNX in onnx_p
        based on https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/python/onnx_custom_plugin/sample.py

        THIS SEEMS TO WORK OK, NO MMDEPLOY STUFF NEEDED?
        (TRT plugin is loaded as .so)
    '''
    # ---- from common -------
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    def GiB(val):
        return val * 1 << 30
    
    MAX_TEXT_LENGTH = 64
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    runtime = trt.Runtime(TRT_LOGGER)

    # Parse model file
    print("Loading ONNX file from path {}...".format(onnx_p))
    with open(onnx_p, "rb") as model:
        print("Beginning ONNX file parsing")
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing of ONNX file")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(4))


     # The input text length is variable, so we need to specify an optimization profile.
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        input = network.get_input(i)
        # assert input.shape[0] == -1     # dynamic input
        min_shape = [1] + list(input.shape[1:])
        opt_shape = [8] + list(input.shape[1:])
        max_shape = [MAX_TEXT_LENGTH] + list(input.shape[1:])
        # profile.set_shape(input.name, min_shape, opt_shape, max_shape)
        profile.set_shape(input.name, min_shape, min_shape, min_shape) # static input
        # TODO: load from a file?

    config.add_optimization_profile(profile)

    print("Building TensorRT engine. This may take a few minutes.")
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    with open(engine_p, "wb") as f:
        f.write(plan)
    
    return engine

def load_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class TRTDetectionModel(DetectionModel):
    def __init__(self, model_path = None, model = None, config_path = None, device = None, 
                 mask_threshold = 0.5, confidence_threshold = 0.3, category_mapping = None, 
                 category_remapping = None, load_at_init = False, image_size = None,
                 ops_plugin_p=None):
        
        super().__init__(model_path, model, config_path, device, mask_threshold, confidence_threshold, 
                         category_mapping, category_remapping, False, image_size)

        self.ops_plugin_p = ops_plugin_p
        self.load_model()

    # TODO:
    #   * load binding names dynamically from a config
    def load_model(self):
        '''
            Args:
                model_p: path to onnx file
                trt_ops_p: custom TRT ops as plugins - shared object file
        '''
        if self.ops_plugin_p is not None:
            ctypes.CDLL(self.ops_plugin_p)

        if Path(self.model_path).suffix == '.onnx':
            engine = load_engine_onnx(self.model_path)
        else:
            engine = load_engine(self.model_path)

        self.set_model(engine)
        self.binding_shapes = get_engine_bindings(engine)

    def set_model(self, model, **kwargs):
        self.model = model

    def set_device(self):
        self.device = 'cuda'

    def unload_model(self):
        print("should be unloading the model")
        # TODO: implement

    def perform_inference(self, image):
        '''
            Args:
                images: ndarray of loaded images: (b, c, h, w)

            Returns:
                    dictionary of ndarrays containing containing batched results e.g.:
                    masks dimensions: (image_idx, detection_idx, height, width)
        '''
  
        # ndarray expected
        input_image = [preprocess(im) for im in image] # this is disgustingly dump!!!
        image_height, image_width = input_image[0].shape[1:]

        engine = self.model

        with self.model.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(engine.get_binding_index("input"), (len(input_image), 3, image_height, image_width))
            # Allocate host and device buffers
            bindings = []

            binding = 'input'
            binding_idx = engine.get_binding_index(binding)
            input_dims = context.get_binding_shape(binding_idx)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_buffer.nbytes)
            bindings.append(int(input_memory))

            binding = 'dets'
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))    
            output_buffer_dets = cuda.pagelocked_empty(size, dtype)
            output_memory_dets = cuda.mem_alloc(output_buffer_dets.nbytes)
            bindings.append(int(output_memory_dets))

            binding = 'labels'
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))    
            output_buffer_labels = cuda.pagelocked_empty(size, dtype)
            output_memory_labels = cuda.mem_alloc(output_buffer_labels.nbytes)
            bindings.append(int(output_memory_labels))

            binding = 'masks'
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))    
            output_buffer_masks = cuda.pagelocked_empty(size, dtype)
            output_memory_masks = cuda.mem_alloc(output_buffer_masks.nbytes)
            bindings.append(int(output_memory_masks))

            stream = cuda.Stream()
            
            # Transfer input data to the GPU.
            t_ingpu = time.monotonic()
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            t_ingpu = time.monotonic() - t_ingpu

            # Run inference
            t_inference = time.monotonic()
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            t_inference = time.monotonic() - t_inference

            # Transfer prediction output from the GPU.
            t_outgpu = time.monotonic()
            cuda.memcpy_dtoh_async(output_buffer_dets, output_memory_dets, stream)
            cuda.memcpy_dtoh_async(output_buffer_labels, output_memory_labels, stream)
            cuda.memcpy_dtoh_async(output_buffer_masks, output_memory_masks, stream)
            # Synchronize the stream
            stream.synchronize()
            t_outgpu = time.monotonic() - t_outgpu

        print("inference done")

        ## postprocessing
        if sum(output_buffer_dets.shape) == 1:
            print("No detections")
            return

        dets = np.reshape(output_buffer_dets, self.binding_shapes['dets'])           
        labels = np.reshape(output_buffer_labels, self.binding_shapes['labels'])
        masks = np.reshape(output_buffer_masks, self.binding_shapes['masks'])

        # upscale the masks from static head output
        masks_global = np.zeros((*masks.shape[:2], *input_dims[2:]), dtype=bool)
        for i in range(len(masks)): # batch
            for j, m in enumerate(masks[i]): 
                box = dets[i, j, :] # batch-1
                x1,y1,x2,y2,score = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # validity check will be implemented within _create_object_predictions_from_original_predictions
                if score < 0.1 or x2-x1 < 3 or y2-y1 < 3: 
                    continue

                dsize = (x2-x1, y2-y1)
                mask_regional = cv2.resize(m, dsize=dsize, interpolation=cv2.INTER_NEAREST) # stretch the mask to cover the region (from 28x28 to bbox size)
                masks_global[i, j, y1:y2, x1:x2] = mask_regional
        
        self._original_predictions = {'dets': dets, 
                                      'labels': labels, 
                                      'masks': masks_global}        

    def _create_object_predictions_from_original_predictions(self, 
                                                             offset_amounts: Optional[List[List[int]]] = [[0, 0]], 
                                                             full_shapes: Optional[List[List[int]]] = None):
        batch_size = len(self._original_predictions['dets'])
        object_predictions_per_image = [None]*batch_size
        for i in range(batch_size):
            dets = self._original_predictions['dets'][i]
            labels = self._original_predictions['labels'][i]
            masks = self._original_predictions['masks'][i]

            object_predictions = [] # keep only valid here
            for j, det in enumerate(dets):
                bbox = det[:4]
                score = det[4]
                label = labels[j]
                mask = masks[j]
                full_shape = None if full_shapes is None else full_shapes[i]

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

                # ignore invalid predictions (too small diff. x2-x1 || y2-y1)
                if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2 or score < 0.15:
                    continue
                
                op = ObjectPrediction(
                    bbox=bbox,
                    score=score,
                    category_id=int(label),
                    bool_mask=mask,
                    offset_amount=offset_amounts[i],
                    full_shape=full_shape
                )
                object_predictions.append(op)
            object_predictions_per_image[i] = object_predictions
        self._object_predictions_per_image = object_predictions_per_image
