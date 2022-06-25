#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

struct OpenCLInfo {
  cl::Context context;
  cl::Platform platform;
  cl::Device device;
  cl::CommandQueue queue;
  cl::Program alpha_removal_program;
  cl::Program::Sources sources;
};

int main() {
  const size_t SIZE = 32;
  OpenCLInfo clinfo;
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    throw std::runtime_error("No OpenCL platforms found");
  }

  clinfo.platform = all_platforms[0];
  printf("Using platform: %s\n",
         clinfo.platform.getInfo<CL_PLATFORM_NAME>().c_str());

  std::vector<cl::Device> all_devices;
  clinfo.platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  clinfo.device = all_devices[0];
  printf("Using OpenCL device: %s\n",
         clinfo.device.getInfo<CL_DEVICE_NAME>().c_str());

  cl::Context ctx({clinfo.device});
  clinfo.context = ctx;

  cl::CommandQueue cmdqueue(clinfo.context, clinfo.device);
  clinfo.queue = cmdqueue;

  // Load kernel
  std::string rgba_to_rgb_kernel_str_code =
      "kernel void cl_rgba_to_rgb(global const char *in, global char *out, "
      "const unsigned int length)"
      "{"
      "	int in_idx = get_global_id(0);"
      "	if(in_idx < length)"
      "	{"
      "		out[in_idx] = in[in_idx];"
      "	}"
      "}";

  clinfo.sources.push_back({rgba_to_rgb_kernel_str_code.c_str(),
                            rgba_to_rgb_kernel_str_code.length()});
  cl::Program program(clinfo.context, clinfo.sources);
  clinfo.alpha_removal_program = program;

  if (clinfo.alpha_removal_program.build({clinfo.device}) != CL_SUCCESS) {
    std::cout << "Error building: "
              << clinfo.alpha_removal_program
                     .getBuildInfo<CL_PROGRAM_BUILD_LOG>(clinfo.device)
              << "\n";
    exit(-1);
  }

  // Tests for kernel
  char *in_h = new char[SIZE];
  char *out_h = new char[SIZE];

  for (uint32_t i = 0; i < SIZE; i++) {
    in_h[i] = (char)i;
  }

  cl::Buffer in_d(clinfo.context, CL_MEM_READ_ONLY, sizeof(char) * SIZE);
  cl::Buffer out_d(clinfo.context, CL_MEM_WRITE_ONLY, sizeof(char) * SIZE);

  clinfo.queue.enqueueWriteBuffer(in_d, CL_TRUE, 0, sizeof(char) * SIZE, in_h);

  cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, const unsigned int>
      cl_rgba_to_rgb(
          cl::Kernel(clinfo.alpha_removal_program, "cl_rgba_to_rgb"));
  cl::NDRange global(SIZE);
  cl_rgba_to_rgb(cl::EnqueueArgs(clinfo.queue, global), in_d, out_d, SIZE)
      .wait();

  clinfo.queue.enqueueReadBuffer(out_d, CL_TRUE, 0, sizeof(char) * SIZE, out_h);

  for (uint32_t i = 0; i < SIZE; i++) {
    printf("%d out_h: %d\n", i, out_h[i]);
  }
}
