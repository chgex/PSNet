

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# import matplotlib.pyplot as plt
from PIL import Image
import os



TRT_LOGGER = trt.Logger()

with engine.create_execution_context() as context:
    # Set input shape based on image dimensions for inference
    context.set_binding_shape(engine.get_binding_index("inputs"), (1, 3, 320, 448))
    # Allocate host and device buffers
    bindings = []
    out_bufs = []
    out_mems = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_image.nbytes)
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))

            out_bufs.append(output_buffer)
            out_mems.append(output_memory)

    stream = cuda.Stream()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer prediction output from the GPU.
    # cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    [cuda.memcpy_dtoh_async(out_buf, out_mem, stream) for out_buf, out_mem in zip(out_bufs, out_mems)]
    # Synchronize the stream
    stream.synchronize()

outputs = [np.reshape(out, (-1, 320, 448)) for out in out_bufs]


class TrtPSNet():
    def __init__(self, trt_path):



    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


class TrtModel():
    def __init__(self, trt_path):
        self.ctx =cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def __call__(self ,img_np_nchw):
        '''
        TensorRT推理
        :param img_np_nchw: 输入图像
        '''
        self.ctx.push()

        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs[0], img_np_nchw.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()
        return host_outputs[0]

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
