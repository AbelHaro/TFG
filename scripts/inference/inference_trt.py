#from ultralytics import YOLO # type: ignore
import tensorrt as trt # type: ignore
import pycuda.driver as cuda # type: ignore
import numpy as np # type: ignore
from collections import OrderedDict, namedtuple
import torch

engine_path = '../../models/canicas/2024_11_28/custom_yolo11.engine'

#print(model.model)

host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

with open(engine_path, 'rb') as f:
    serialized_engine = f.read()
    
runtime = trt.Runtime(TRT_LOGGER)

model = runtime.deserialize_cuda_engine(serialized_engine)

try:
    context = model.create_execution_context()
except Exception as e:  # model is None
    LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
    raise e


bindings = OrderedDict()
output_names = []
fp16 = False  # default updated below
dynamic = False
is_trt10 = not hasattr(model, "num_bindings")
device = torch.device("cuda:0")
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
for i in num:
    name = model.get_binding_name(i)
    dtype = trt.nptype(model.get_binding_dtype(i))
    is_input = model.binding_is_input(i)
    if model.binding_is_input(i):
        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
            dynamic = True
            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
        if dtype == np.float16:
            fp16 = True
    else:
        output_names.append(name)
    shape = tuple(context.get_binding_shape(i))
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
batch_size = 1
    
    