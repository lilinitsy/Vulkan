/*
 * Vulkan Example - Scene rendering
 *
 * Copyright (C) 2020 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 *
 * Summary:
 * Render a complete scene loaded from an glTF file. The sample is based on the
 * glTF model loading sample, and adds data structures, functions and shaders
 * required to render a more complex scene using Crytek's Sponza model.
 *
 * This sample comes with a tutorial, see the README.md in this folder
 */


#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_beta.h>

#include <sys/time.h>

extern "C"
{
	#include <libavcodec/avcodec.h>
	#include <libswscale/swscale.h>		
	#include <libavutil/opt.h>
	#include <libavutil/imgutils.h>
}

//#include <ffnvcodec/nvEncodeAPI.h>

#include <CL/opencl.hpp>


#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif


#include "tiny_gltf.h"

#include "imagepacket.h"
#include "server.h"
#include "vk_utils.h"
#include "vulkanexamplebase.h"

#define ENABLE_VALIDATION true


const uint32_t PORT[2] = {1234, 1235};

// Offloaded rendering attributes
const uint32_t SERVERWIDTH  = 2400; // 512
const uint32_t SERVERHEIGHT = 1080; // 512
const uint32_t CLIENTWIDTH  = 2400;
const uint32_t CLIENTHEIGHT = 1080;

// Possibly temp offloaded rendering attributes
const uint32_t FOVEAWIDTH  = 480;
const uint32_t FOVEAHEIGHT = 272;


// Contains everything required to render a basic glTF scene in Vulkan
// This class is heavily simplified (compared to glTF's feature set) but retains
// the basic glTF structure
class VulkanglTFScene
{
  public:
	// The class requires some Vulkan objects so it can create it's own resources
	vks::VulkanDevice *vulkanDevice;
	VkQueue copyQueue;

	// The vertex layout for the samples' model
	struct Vertex
	{
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec3 color;
		glm::vec4 tangent;
	};

	// Single vertex buffer for all primitives
	struct
	{
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertices;

	// Single index buffer for all primitives
	struct
	{
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;

	// The following structures roughly represent the glTF scene structure
	// To keep things simple, they only contain those properties that are required
	// for this sample
	struct Node;

	// A primitive contains the data for a single draw call
	struct Primitive
	{
		uint32_t firstIndex;
		uint32_t indexCount;
		int32_t materialIndex;
	};

	// Contains the node's (optional) geometry and can be made up of an arbitrary
	// number of primitives
	struct Mesh
	{
		std::vector<Primitive> primitives;
	};

	// A node represents an object in the glTF scene graph
	struct Node
	{
		Node *parent;
		std::vector<Node> children;
		Mesh mesh;
		glm::mat4 matrix;
		std::string name;
		bool visible = true;
	};

	// A glTF material stores information in e.g. the texture that is attached to
	// it and colors
	struct Material
	{
		glm::vec4 baseColorFactor = glm::vec4(1.0f);
		uint32_t baseColorTextureIndex;
		uint32_t normalTextureIndex;
		std::string alphaMode = "OPAQUE";
		float alphaCutOff;
		bool doubleSided = false;
		VkDescriptorSet descriptorSet;
		VkPipeline pipeline;
	};

	// Contains the texture for a single glTF image
	// Images may be reused by texture objects and are as such separated
	struct Image
	{
		vks::Texture2D texture;
	};

	// A glTF texture stores a reference to the image and a sampler
	// In this sample, we are only interested in the image
	struct Texture
	{
		int32_t imageIndex;
	};

	/*
          Model data
  */
	std::vector<Image> images;
	std::vector<Texture> textures;
	std::vector<Material> materials;
	std::vector<Node> nodes;

	std::string path;

	~VulkanglTFScene();
	VkDescriptorImageInfo getTextureDescriptor(const size_t index);
	void loadImages(tinygltf::Model &input);
	void loadTextures(tinygltf::Model &input);
	void loadMaterials(tinygltf::Model &input);
	void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input,
	              VulkanglTFScene::Node *parent,
	              std::vector<uint32_t> &indexBuffer,
	              std::vector<VulkanglTFScene::Vertex> &vertexBuffer);
	void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
	              VulkanglTFScene::Node node);
	void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout);
};


class VulkanExample : public VulkanExampleBase
{
  public:
	VulkanglTFScene glTFScene;


	struct ShaderData
	{
		vks::Buffer buffer;

		// projection and view will each need to be 2 element arrays
		struct Values
		{
			glm::mat4 projection[2];
			glm::mat4 view[2];
			glm::vec4 lightPos = glm::vec4(0.0f, 2.5f, 0.0f, 1.0f);
			glm::vec4 viewPos;
		} values;
	} shaderData;


	struct
	{
		VkPipelineLayout multiview;
		VkPipelineLayout viewdisp;
	} pipeline_layouts;

	struct
	{
		VkDescriptorSet multiview;
		VkDescriptorSet viewdisp;
	} descriptor_sets;


	struct
	{
		VkDescriptorSetLayout matrices;
		VkDescriptorSetLayout textures;
		VkDescriptorSetLayout viewdisp;
	} descriptor_set_layouts;

	// Imported stuff from multiview/multiview.cpp
	struct FrameBufferAttachment
	{
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
	};

	VkPipeline viewdisp_pipelines[2];
	VkPipeline material_pipeline;

	struct MultiviewPass
	{
		FrameBufferAttachment colour;
		FrameBufferAttachment depth;
		VkFramebuffer framebuffer;
		VkRenderPass renderpass;
		VkDescriptorImageInfo descriptor;
		VkSampler sampler;
		VkSemaphore semaphore;
		std::vector<VkCommandBuffer> command_buffers;
		std::vector<VkFence> wait_fences;
	} multiview_pass;

	// ImagePackets to copy shit to
	ImagePacket lefteye_fovea;
	ImagePacket righteye_fovea;

	ImagePacket foveal_regions;

	Server server;

	struct
	{
		std::vector<float> drawtime;
		std::vector<float> copy_image_time;
		std::vector<float> remove_alpha_time;
	} timers;

	struct
	{
		float left_remove_alpha_time;
		float right_remove_alpha_time;
	} tmp_timers;


	struct
	{
		pthread_t send_image;
		pthread_t recv_camera;
	} vk_pthread;

	VkPhysicalDeviceMultiviewFeaturesKHR physical_device_multiview_features{};

	int numframes = 0;


	// Camera and view properties
	float eyeSeparation     = 0.08f;
	const float focalLength = 0.5f;
	const float fov         = 90.0f;
	const float zNear       = 0.1f;
	const float zFar        = 256.0f;

	bool enable_multiview = true;

	struct
	{
		cl::Context context;
		cl::Platform platform;
		cl::Device device;
		cl::CommandQueue queue;
		cl::Program alpha_removal_program;
		cl::Program::Sources sources;
	} cl;

	struct
	{
		const AVCodec *codec;
		AVCodecContext *c;
		AVFrame *frame;
		AVPacket *packet;
	} encoder;


	struct
	{
		const AVCodec *codec;
		AVCodecContext *c;
		AVCodecParserContext *parser;
		AVFrame *frame;
		AVPacket *packet;
	} decoder;

	bool should_wait_for_camera_data = true;

	


	VulkanExample();
	~VulkanExample();
	virtual void getEnabledFeatures();
	void buildCommandBuffers();
	void loadglTFFile(std::string filename);
	void loadAssets();
	void setupDescriptors();
	void preparePipelines();
	void prepareUniformBuffers();
	void updateUniformBuffers();
	void prepare();
	void draw();

	void setup_multiview();
	ImagePacket create_image_packet();
	ImagePacket copy_image_to_packet(VkImage src_image, ImagePacket image_packet, VkOffset3D left_offset, VkOffset3D right_offset);
	void write_imagepacket_to_file(ImagePacket packet, uint32_t buffer, std::string name);

	void setup_video_encoder();
	//void begin_video_encoding(uint8_t *luminance_y, uint8_t *bp_u, uint8_t *rp_v);
	//void encode(AVCodecContext *encode_context, AVFrame *frame, AVPacket *packet, FILE *outfile);


	void setup_video_decoder();
	void begin_video_decoding();
	void decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt, const char *filename);


	void setup_opencl();
	void rgba_to_rgb_opencl(const uint8_t *__restrict__ in_h, uint8_t *__restrict__ out_Y_h, uint8_t *__restrict__ out_U_h, uint8_t *__restrict__ out_V_h, size_t len);

	virtual void render();
	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay);
};