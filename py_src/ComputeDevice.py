"""
    This class handles all OpenCL code.
    Launches n worker threads, each of which sums up a different subseries.
    The host sends each consecutive term to the compute device, each worker then decides whether it should add the given term.
    We can either send the device a list of terms to add or send them in one by one.
"""
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

class ComputeDevice:
    """
    *******************************************
        Set-Up Methods
    *******************************************
    """
    def __init__(self, k):
        self.k = k
        self.iterations = 0
        self.worker_count = 2**self.k
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        self.create_host_buffers() # Create data on host.
        self.create_device_buffers() # Send data to device.
        self.create_programs() # Compile device programs.
        self.setup_points() # Shift points to (1,0).

    def create_host_buffers(self):
        """
        Create arrays which will be sent to compute device.
        """
        self.codes_np = np.arange(0, self.worker_count, dtype=np.uint32)
        self.switches_np = np.zeros(self.worker_count, dtype=np.uint32)
        self.points_np = np.zeros(self.worker_count, dtype=cl_array.vec.float2)

    def create_device_buffers(self):
        """
        Send the data to the compute device.
        """
        mf = cl.mem_flags
        self.code_master_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.codes_np)
        self.code_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.codes_np)
        self.switch_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.switches_np)
        self.point_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.points_np)

    def create_programs(self):
        source_file = open('codes.cl')
        source = source_file.read()
        source_file.close()
        self.codes_program = cl.Program(self.context, source).build()

        source_file = open('main.cl')
        source = source_file.read()
        source_file.close()
        self.main_program = cl.Program(self.context, source).build()

    def setup_points(self):
        """
        Points start off at the origin, so we shift them to (1,0).
        """
        source_file = open('setup.cl')
        source = source_file.read()
        source_file.close()
        program = cl.Program(self.context, source).build()
        program.setup_points(self.queue, self.points_np.shape, None, self.point_buffer)


    """
    *******************************************
        Buffer Getter Methods
    *******************************************
    """
    def get_point_buffer(self):
        results = np.empty_like(self.points_np)
        cl.enqueue_copy(self.queue, results, self.point_buffer)
        return results

    def get_switch_buffer(self):
        results = np.empty_like(self.switches_np)
        cl.enqueue_copy(self.queue, results, self.switch_buffer)
        return results

    def get_code_buffer(self):
        results = np.empty_like(self.codes_np)
        cl.enqueue_copy(self.queue, results, self.code_buffer)
        return results


    """
    *******************************************
        Print Methods
    *******************************************
    """
    def print_point_buffer(self):
        print(self.get_point_buffer())

    def print_switch_buffer(self):
        print(self.get_switch_buffer())

    def print_code_buffer(self):
        print(self.get_code_buffer())


    """
    *******************************************
        Kernel Methods
    *******************************************
    """
    def codes_reset(self):
        self.codes_program.reset_codes(self.queue, self.codes_np.shape, None, self.code_buffer, self.code_master_buffer)

    def codes_update(self):
        self.codes_program.update_codes(self.queue, self.codes_np.shape, None, self.code_buffer)

    def codes_decode(self):
        self.codes_program.decode(self.queue, self.switches_np.shape, None, self.switch_buffer, self.code_buffer)

    def main_update_points(self, term):
        re = np.float32(term.real)
        im = np.float32(term.imag)
        self.main_program.update_points(self.queue, self.points_np.shape, None, self.point_buffer, re, im, self.switch_buffer)

    """
    *******************************************
        End User Methods
    *******************************************
    """
    def process_next_term(self, term):
        if ((self.iterations % self.k) == 0 ):
            self.codes_reset()
        self.codes_decode()
        self.main_update_points(term)
        self.codes_update()
        self.iterations += 1

    """
    *******************************************
        Debug Methods
    *******************************************
    """
    def update_switches(self):
        if ((self.iterations % self.k) == 0 ):
            self.codes_reset()
        self.codes_decode()
        self.codes_update()
        self.print_switch_buffer()
        self.iterations += 1
