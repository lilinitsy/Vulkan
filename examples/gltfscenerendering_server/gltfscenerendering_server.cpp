/*
 * Vulkan Example - Scene rendering
 *
 * Copyright (C) 2020-2021 by Sascha Willems - www.saschawillems.de
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


#include "gltfscenerendering_server.h"
#include "vk_utils.h"
#include "vulkan/vulkan_beta.h"
#include "vulkan/vulkan_core.h"
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <stdexcept>
#include <string>


/*
        Vulkan glTF scene class
*/

VulkanglTFScene::~VulkanglTFScene()
{
	// Release all Vulkan resources allocated for the model
	vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
	vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
	vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
	vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
	for(Image image : images)
	{
		vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view,
		                   nullptr);
		vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
		vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler,
		                 nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory,
		             nullptr);
	}
	for(Material material : materials)
	{
		vkDestroyPipeline(vulkanDevice->logicalDevice, material.pipeline, nullptr);
	}
}

/*
        glTF loading functions

        The following functions take a glTF input model loaded via tinyglTF and
   convert all required data into our own structure
*/

void VulkanglTFScene::loadImages(tinygltf::Model &input)
{
	// POI: The textures for the glTF file used in this sample are stored as
	// external ktx files, so we can directly load them from disk without the need
	// for conversion
	images.resize(input.images.size());
	for(size_t i = 0; i < input.images.size(); i++)
	{
		tinygltf::Image &glTFImage = input.images[i];
		images[i].texture.loadFromFile(path + "/" + glTFImage.uri,
		                               VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice,
		                               copyQueue);
	}
}

void VulkanglTFScene::loadTextures(tinygltf::Model &input)
{
	textures.resize(input.textures.size());
	for(size_t i = 0; i < input.textures.size(); i++)
	{
		textures[i].imageIndex = input.textures[i].source;
	}
}

void VulkanglTFScene::loadMaterials(tinygltf::Model &input)
{
	materials.resize(input.materials.size());
	for(size_t i = 0; i < input.materials.size(); i++)
	{
		// We only read the most basic properties required for our sample
		tinygltf::Material glTFMaterial = input.materials[i];
		// Get the base color factor
		if(glTFMaterial.values.find("baseColorFactor") !=
		   glTFMaterial.values.end())
		{
			materials[i].baseColorFactor = glm::make_vec4(
				glTFMaterial.values["baseColorFactor"].ColorFactor().data());
		}
		// Get base color texture index
		if(glTFMaterial.values.find("baseColorTexture") !=
		   glTFMaterial.values.end())
		{
			materials[i].baseColorTextureIndex =
				glTFMaterial.values["baseColorTexture"].TextureIndex();
		}
		// Get the normal map texture index
		if(glTFMaterial.additionalValues.find("normalTexture") !=
		   glTFMaterial.additionalValues.end())
		{
			materials[i].normalTextureIndex =
				glTFMaterial.additionalValues["normalTexture"].TextureIndex();
		}
		// Get some additional material parameters that are used in this sample
		materials[i].alphaMode   = glTFMaterial.alphaMode;
		materials[i].alphaCutOff = (float) glTFMaterial.alphaCutoff;
		materials[i].doubleSided = glTFMaterial.doubleSided;
	}
}

void VulkanglTFScene::loadNode(
	const tinygltf::Node &inputNode, const tinygltf::Model &input,
	VulkanglTFScene::Node *parent, std::vector<uint32_t> &indexBuffer,
	std::vector<VulkanglTFScene::Vertex> &vertexBuffer)
{
	VulkanglTFScene::Node node{};
	node.name = inputNode.name;

	// Get the local node matrix
	// It's either made up from translation, rotation, scale or a 4x4 matrix
	node.matrix = glm::mat4(1.0f);
	if(inputNode.translation.size() == 3)
	{
		node.matrix = glm::translate(
			node.matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
	}
	if(inputNode.rotation.size() == 4)
	{
		glm::quat q = glm::make_quat(inputNode.rotation.data());
		node.matrix *= glm::mat4(q);
	}
	if(inputNode.scale.size() == 3)
	{
		node.matrix = glm::scale(node.matrix,
		                         glm::vec3(glm::make_vec3(inputNode.scale.data())));
	}
	if(inputNode.matrix.size() == 16)
	{
		node.matrix = glm::make_mat4x4(inputNode.matrix.data());
	};

	// Load node's children
	if(inputNode.children.size() > 0)
	{
		for(size_t i = 0; i < inputNode.children.size(); i++)
		{
			loadNode(input.nodes[inputNode.children[i]], input, &node, indexBuffer,
			         vertexBuffer);
		}
	}

	// If the node contains mesh data, we load vertices and indices from the
	// buffers In glTF this is done via accessors and buffer views
	if(inputNode.mesh > -1)
	{
		const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
		// Iterate through all primitives of this node's mesh
		for(size_t i = 0; i < mesh.primitives.size(); i++)
		{
			const tinygltf::Primitive &glTFPrimitive = mesh.primitives[i];
			uint32_t firstIndex                      = static_cast<uint32_t>(indexBuffer.size());
			uint32_t vertexStart                     = static_cast<uint32_t>(vertexBuffer.size());
			uint32_t indexCount                      = 0;
			// Vertices
			{
				const float *positionBuffer  = nullptr;
				const float *normalsBuffer   = nullptr;
				const float *texCoordsBuffer = nullptr;
				const float *tangentsBuffer  = nullptr;
				size_t vertexCount           = 0;

				// Get buffer data for vertex normals
				if(glTFPrimitive.attributes.find("POSITION") !=
				   glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor =
						input
							.accessors[glTFPrimitive.attributes.find("POSITION")->second];
					const tinygltf::BufferView &view =
						input.bufferViews[accessor.bufferView];
					positionBuffer = reinterpret_cast<const float *>(
						&(input.buffers[view.buffer]
					          .data[accessor.byteOffset + view.byteOffset]));
					vertexCount = accessor.count;
				}
				// Get buffer data for vertex nfavrmals
				if(glTFPrimitive.attributes.find("NORMAL") !=
				   glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor =
						input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
					const tinygltf::BufferView &view =
						input.bufferViews[accessor.bufferView];
					normalsBuffer = reinterpret_cast<const float *>(
						&(input.buffers[view.buffer]
					          .data[accessor.byteOffset + view.byteOffset]));
				}
				// Get buffer data for vertex texture coordinates
				// glTF supports multiple sets, we only load the first one
				if(glTFPrimitive.attributes.find("TEXCOORD_0") !=
				   glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor =
						input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")
					                        ->second];
					const tinygltf::BufferView &view =
						input.bufferViews[accessor.bufferView];
					texCoordsBuffer = reinterpret_cast<const float *>(
						&(input.buffers[view.buffer]
					          .data[accessor.byteOffset + view.byteOffset]));
				}
				// POI: This sample uses normal mapping, so we also need to load the
				// tangents from the glTF file
				if(glTFPrimitive.attributes.find("TANGENT") !=
				   glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor =
						input.accessors[glTFPrimitive.attributes.find("TANGENT")->second];
					const tinygltf::BufferView &view =
						input.bufferViews[accessor.bufferView];
					tangentsBuffer = reinterpret_cast<const float *>(
						&(input.buffers[view.buffer]
					          .data[accessor.byteOffset + view.byteOffset]));
				}

				// Append data to model's vertex buffer
				for(size_t v = 0; v < vertexCount; v++)
				{
					Vertex vert{};
					vert.pos    = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
					vert.normal = glm::normalize(
						glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
					vert.uv      = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
					vert.color   = glm::vec3(1.0f);
					vert.tangent = tangentsBuffer ? glm::make_vec4(&tangentsBuffer[v * 4]) : glm::vec4(0.0f);
					vertexBuffer.push_back(vert);
				}
			}
			// Indices
			{
				const tinygltf::Accessor &accessor =
					input.accessors[glTFPrimitive.indices];
				const tinygltf::BufferView &bufferView =
					input.bufferViews[accessor.bufferView];
				const tinygltf::Buffer &buffer = input.buffers[bufferView.buffer];

				indexCount += static_cast<uint32_t>(accessor.count);

				// glTF supports different component types of indices
				switch(accessor.componentType)
				{
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
					{
						const uint32_t *buf = reinterpret_cast<const uint32_t *>(
							&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for(size_t index = 0; index < accessor.count; index++)
						{
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
					{
						const uint16_t *buf = reinterpret_cast<const uint16_t *>(
							&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for(size_t index = 0; index < accessor.count; index++)
						{
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
					{
						const uint8_t *buf = reinterpret_cast<const uint8_t *>(
							&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for(size_t index = 0; index < accessor.count; index++)
						{
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					default:
						std::cerr << "Index component type " << accessor.componentType
								  << " not supported!" << std::endl;
						return;
				}
			}
			Primitive primitive{};
			primitive.firstIndex    = firstIndex;
			primitive.indexCount    = indexCount;
			primitive.materialIndex = glTFPrimitive.material;
			node.mesh.primitives.push_back(primitive);
		}
	}

	if(parent)
	{
		parent->children.push_back(node);
	}
	else
	{
		nodes.push_back(node);
	}
}

VkDescriptorImageInfo
VulkanglTFScene::getTextureDescriptor(const size_t index)
{
	return images[index].texture.descriptor;
}

/*
        glTF rendering functions
*/

// Draw a single node including child nodes (if present)
void VulkanglTFScene::drawNode(VkCommandBuffer commandBuffer,
                               VkPipelineLayout pipelineLayout,
                               VulkanglTFScene::Node node)
{
	if(!node.visible)
	{
		return;
	}
	if(node.mesh.primitives.size() > 0)
	{
		// Pass the node's matrix via push constants
		// Traverse the node hierarchy to the top-most parent to get the final
		// matrix of the current node
		glm::mat4 nodeMatrix                 = node.matrix;
		VulkanglTFScene::Node *currentParent = node.parent;
		while(currentParent)
		{
			nodeMatrix    = currentParent->matrix * nodeMatrix;
			currentParent = currentParent->parent;
		}
		// Pass the final matrix to the vertex shader using push constants
		vkCmdPushConstants(commandBuffer, pipelineLayout,
		                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
		                   &nodeMatrix);
		for(VulkanglTFScene::Primitive &primitive : node.mesh.primitives)
		{
			if(primitive.indexCount > 0)
			{
				VulkanglTFScene::Material &material =
					materials[primitive.materialIndex];
				// POI: Bind the pipeline for the node's material
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				                  material.pipeline);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				                        pipelineLayout, 1, 1, &material.descriptorSet,
				                        0, nullptr);

				vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1,
				                 primitive.firstIndex, 0, 0);
			}
		}
	}
	for(auto &child : node.children)
	{
		drawNode(commandBuffer, pipelineLayout, child);
	}
}

// Draw the glTF scene starting at the top-level-nodes
void VulkanglTFScene::draw(VkCommandBuffer commandBuffer,
                           VkPipelineLayout pipelineLayout)
{
	// All vertices and indices are stored in single buffers, so we only need to
	// bind once
	VkDeviceSize offsets[1] = {0};
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
	// Render all nodes at top-level
	for(auto &node : nodes)
	{
		drawNode(commandBuffer, pipelineLayout, node);
	}
}

/*
        Vulkan Example class
*/

VulkanExample::VulkanExample() :
	VulkanExampleBase(ENABLE_VALIDATION, SERVERWIDTH, SERVERHEIGHT)
{
	title        = "glTF scene rendering";
	camera.type  = Camera::CameraType::firstperson;
	camera.flipY = true;
	camera.setPosition(glm::vec3(2.2f, -2.0f, 0.25f));
	camera.setRotation(glm::vec3(-180.0f, -90.0f, 0.0f));
	camera.movementSpeed = 4.0f;

	printf("Scene width, height: %d\t%d\n", width, height);
	printf("Fullscreen: %d\n", settings.fullscreen);

	// Multiview setup
	// Enable extension required for multiview
	enabledDeviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);

	// Reading device properties and features for multiview requires VK_KHR_get_physical_device_properties2 to be enabled
	enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

	// Video requirements
	enabledDeviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
	enabledDeviceExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);

	// Dependences for YCbCr_Conversion
	//VK_KHR_maintenance1, VK_KHR_bind_memory2, VK_KHR_get_memory_requirements2
	enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
	enabledDeviceExtensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
	enabledDeviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);

	// Enable required extension features
	physical_device_multiview_features = {
		.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR,
		.multiview = VK_TRUE,
	};
	deviceCreatepNextChain = &physical_device_multiview_features;
}

VulkanExample::~VulkanExample()
{
	vkDestroyPipelineLayout(device, pipeline_layouts.multiview, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptor_set_layouts.matrices, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptor_set_layouts.textures, nullptr);
	shaderData.buffer.destroy();

	// Multiview destroyers
	vkDestroyImageView(device, multiview_pass.colour.view, nullptr);
	vkDestroyImage(device, multiview_pass.colour.image, nullptr);
	vkFreeMemory(device, multiview_pass.colour.memory, nullptr);
	vkDestroyImageView(device, multiview_pass.depth.view, nullptr);
	vkDestroyImage(device, multiview_pass.depth.image, nullptr);
	vkFreeMemory(device, multiview_pass.depth.memory, nullptr);

	vkDestroyRenderPass(device, multiview_pass.renderpass, nullptr);
	vkDestroySampler(device, multiview_pass.sampler, nullptr);
	vkDestroyFramebuffer(device, multiview_pass.framebuffer, nullptr);

	vkDestroySemaphore(device, multiview_pass.semaphore, nullptr);
	for(uint32_t i = 0; i < multiview_pass.wait_fences.size(); i++)
	{
		vkDestroyFence(device, multiview_pass.wait_fences[i], nullptr);
	}



}

void VulkanExample::getEnabledFeatures()
{
	enabledFeatures.samplerAnisotropy = deviceFeatures.samplerAnisotropy;
}


void VulkanExample::setup_multiview()
{
	uint32_t multiview_layers = 2;

	// Colour attachment setup
	{
		VkImageCreateInfo image_ci = {
			.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext         = nullptr,
			.flags         = 0,
			.imageType     = VK_IMAGE_TYPE_2D,
			.format        = swapChain.colorFormat,
			.extent        = {width, height, 1},
			.mipLevels     = 1,
			.arrayLayers   = multiview_layers,
			.samples       = VK_SAMPLE_COUNT_1_BIT,
			.tiling        = VK_IMAGE_TILING_OPTIMAL,
			.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		VK_CHECK_RESULT(vkCreateImage(device, &image_ci, nullptr, &multiview_pass.colour.image));


		VkMemoryRequirements memory_requirements;
		vkGetImageMemoryRequirements(device, multiview_pass.colour.image, &memory_requirements);

		VkMemoryAllocateInfo memory_ai = {
			.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize  = memory_requirements.size,
			.memoryTypeIndex = vulkanDevice->getMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
		};
		VK_CHECK_RESULT(vkAllocateMemory(device, &memory_ai, nullptr, &multiview_pass.colour.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, multiview_pass.colour.image, multiview_pass.colour.memory, 0));


		VkImageSubresourceRange colour_subresource = {
			.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = multiview_layers,
		};

		VkImageViewCreateInfo image_view_ci = {
			.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext            = nullptr,
			.flags            = 0,
			.image            = multiview_pass.colour.image,
			.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			.format           = swapChain.colorFormat,
			.subresourceRange = colour_subresource,
		};
		VK_CHECK_RESULT(vkCreateImageView(device, &image_view_ci, nullptr, &multiview_pass.colour.view));


		VkSamplerCreateInfo sampler_ci = {
			.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.pNext            = nullptr,
			.flags            = 0,
			.magFilter        = VK_FILTER_NEAREST,
			.minFilter        = VK_FILTER_NEAREST,
			.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias       = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy    = 1.0f,
			.minLod           = 0.0f,
			.maxLod           = 1.0f,
			.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		};
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler_ci, nullptr, &multiview_pass.sampler));

		// Setup the descriptors for the colour attachment
		multiview_pass.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		multiview_pass.descriptor.imageView   = multiview_pass.colour.view;
		multiview_pass.descriptor.sampler     = multiview_pass.sampler;
	}

	// depth/stencil FBO setup
	{
		VkImageCreateInfo image_ci = {
			.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext       = nullptr,
			.flags       = 0,
			.imageType   = VK_IMAGE_TYPE_2D,
			.format      = depthFormat,
			.extent      = {width, height, 1},
			.mipLevels   = 1,
			.arrayLayers = multiview_layers,
			.samples     = VK_SAMPLE_COUNT_1_BIT,
			.tiling      = VK_IMAGE_TILING_OPTIMAL,
			.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		};
		VK_CHECK_RESULT(vkCreateImage(device, &image_ci, nullptr, &multiview_pass.depth.image));


		VkMemoryRequirements memory_requirements;
		vkGetImageMemoryRequirements(device, multiview_pass.depth.image, &memory_requirements);

		VkMemoryAllocateInfo memory_ai = {
			.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize  = memory_requirements.size,
			.memoryTypeIndex = vulkanDevice->getMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
		};


		VkImageSubresourceRange depth_stencil_subresource = {
			.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = multiview_layers,
		};

		VkImageViewCreateInfo depth_stencil_view = {
			.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext            = nullptr,
			.flags            = 0,
			.image            = multiview_pass.depth.image,
			.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			.format           = depthFormat,
			.subresourceRange = depth_stencil_subresource,
		};


		VK_CHECK_RESULT(vkAllocateMemory(device, &memory_ai, nullptr, &multiview_pass.depth.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, multiview_pass.depth.image, multiview_pass.depth.memory, 0));
		VK_CHECK_RESULT(vkCreateImageView(device, &depth_stencil_view, nullptr, &multiview_pass.depth.view));
	}

	// Multiview Renderpass
	{
		VkAttachmentDescription attachments[2];

		// Colour attachment
		attachments[0] = {
			.flags          = 0,
			.format         = swapChain.colorFormat,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		// Depth attachment
		attachments[1] = {
			.flags          = 0,
			.format         = depthFormat,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		// Attachment references
		VkAttachmentReference colour_reference = {
			.attachment = 0,
			.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_reference = {
			.attachment = 1,
			.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};


		// Subpass dependencies
		VkSubpassDescription subpass_description = {
			.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount    = 1,
			.pColorAttachments       = &colour_reference,
			.pDepthStencilAttachment = &depth_reference,
		};

		VkSubpassDependency dependencies[2];
		dependencies[0] = {
			.srcSubpass      = VK_SUBPASS_EXTERNAL,
			.dstSubpass      = 0,
			.srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT,
			.dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
		};

		dependencies[1] = {
			.srcSubpass      = 0,
			.dstSubpass      = VK_SUBPASS_EXTERNAL,
			.srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
		};

		// Bit mask for which view is rendering
		const uint32_t viewmask         = 0b00000011;
		const uint32_t correlation_mask = 0b00000011;

		// Multiview renderpass creation
		VkRenderPassMultiviewCreateInfo renderpass_multiview_ci = {
			.sType                = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
			.pNext                = nullptr,
			.subpassCount         = 1,
			.pViewMasks           = &viewmask,
			.correlationMaskCount = 1,
			.pCorrelationMasks    = &correlation_mask,
		};

		VkRenderPassCreateInfo renderpass_ci = {
			.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.pNext           = &renderpass_multiview_ci,
			.flags           = 0,
			.attachmentCount = 2,
			.pAttachments    = attachments,
			.subpassCount    = 1,
			.pSubpasses      = &subpass_description,
			.dependencyCount = 2,
			.pDependencies   = dependencies,
		};

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderpass_ci, nullptr, &multiview_pass.renderpass));
	}


	// Framebuffer creation
	{
		VkImageView fbo_attachments[] = {multiview_pass.colour.view, multiview_pass.depth.view};

		VkFramebufferCreateInfo fbo_ci = {
			.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.pNext           = nullptr,
			.flags           = 0,
			.renderPass      = multiview_pass.renderpass,
			.attachmentCount = 2,
			.pAttachments    = fbo_attachments,
			.width           = SERVERWIDTH,
			.height          = SERVERHEIGHT,
			.layers          = 1,
		};
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbo_ci, nullptr, &multiview_pass.framebuffer));
	}
}


void VulkanExample::buildCommandBuffers()
{
	int32_t left_mid_boundary = CLIENTWIDTH / 4 - FOVEAWIDTH / 2;
	int32_t top_boundary      = CLIENTHEIGHT / 2 - FOVEAHEIGHT / 2;
	int32_t bottom_boundary   = CLIENTHEIGHT / 2 + FOVEAHEIGHT / 2;

	// View display rendering
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color        = {0.0f, 0.0f, 0.0f, 1.0f}; //defaultClearColor;
		clearValues[1].depthStencil = {1.0f, 0};


		VkRenderPassBeginInfo renderPassBeginInfo    = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass               = renderPass;
		renderPassBeginInfo.renderArea.offset.x      = 0;
		renderPassBeginInfo.renderArea.offset.y      = 0;
		renderPassBeginInfo.renderArea.extent.width  = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount          = 2;
		renderPassBeginInfo.pClearValues             = clearValues;

		for(int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float) width / 2.0f, (float) height, 0.0f, 1.0f);
			VkRect2D scissor    = vks::initializers::rect2D(FOVEAWIDTH, FOVEAHEIGHT, left_mid_boundary, top_boundary);

			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			// Bind descriptor set
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.viewdisp, 0, 1, &descriptor_sets.viewdisp, 0, nullptr);

			// Left eye
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, viewdisp_pipelines[0]);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			// Right eye
			viewport.x = (float) width / 2.0f;
			scissor.offset.x += width / 2.0f;
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, viewdisp_pipelines[1]);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			// DO NOT drawUI in the multiview pass.
			//drawUI(drawCmdBuffers[i]);
			vkCmdEndRenderPass(drawCmdBuffers[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}


	// Multiview GLTF rendering
	{
		multiview_pass.command_buffers.resize(drawCmdBuffers.size());
		VkCommandBufferAllocateInfo cmdbuf_ai = vks::initializers::commandBufferAllocateInfo(cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, static_cast<uint32_t>(drawCmdBuffers.size()));
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdbuf_ai, multiview_pass.command_buffers.data()));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color        = {0.0f, 0.0f, 0.0f, 1.0f};
		clearValues[1].depthStencil = {1.0f, 0};

		VkRenderPassBeginInfo renderPassBeginInfo    = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass               = multiview_pass.renderpass;
		renderPassBeginInfo.renderArea.offset.x      = 0;
		renderPassBeginInfo.renderArea.offset.y      = 0;
		renderPassBeginInfo.renderArea.extent.width  = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount          = 2;
		renderPassBeginInfo.pClearValues             = clearValues;

		const VkViewport viewport = vks::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
		const VkRect2D scissor    = vks::initializers::rect2D(width, height, 0, 0);

		for(int32_t i = 0; i < multiview_pass.command_buffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = multiview_pass.framebuffer;
			VK_CHECK_RESULT(vkBeginCommandBuffer(multiview_pass.command_buffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(multiview_pass.command_buffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdSetViewport(multiview_pass.command_buffers[i], 0, 1, &viewport);
			vkCmdSetScissor(multiview_pass.command_buffers[i], 0, 1, &scissor);

			// Bind scene matrices descriptor to set 0
			vkCmdBindDescriptorSets(multiview_pass.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.multiview, 0, 1, &descriptor_sets.multiview, 0, nullptr);

			// POI: Draw the glTF scene
			glTFScene.draw(multiview_pass.command_buffers[i], pipeline_layouts.multiview);

			vkCmdEndRenderPass(multiview_pass.command_buffers[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(multiview_pass.command_buffers[i]));
		}
	}
}

void VulkanExample::loadglTFFile(std::string filename)
{
	tinygltf::Model glTFInput;
	tinygltf::TinyGLTF gltfContext;
	std::string error, warning;

	this->device = device;

#if defined(__ANDROID__)
	// On Android all assets are packed with the apk in a compressed form, so we
	// need to open them using the asset manager We let tinygltf handle this, by
	// passing the asset manager of our app
	tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
	bool fileLoaded =
		gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

	// Pass some Vulkan resources required for setup and rendering to the glTF
	// model loading class
	glTFScene.vulkanDevice = vulkanDevice;
	glTFScene.copyQueue    = queue;

	size_t pos     = filename.find_last_of('/');
	glTFScene.path = filename.substr(0, pos);

	std::vector<uint32_t> indexBuffer;
	std::vector<VulkanglTFScene::Vertex> vertexBuffer;

	if(fileLoaded)
	{
		glTFScene.loadImages(glTFInput);
		glTFScene.loadMaterials(glTFInput);
		glTFScene.loadTextures(glTFInput);
		const tinygltf::Scene &scene = glTFInput.scenes[0];
		for(size_t i = 0; i < scene.nodes.size(); i++)
		{
			const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
			glTFScene.loadNode(node, glTFInput, nullptr, indexBuffer, vertexBuffer);
		}
	}
	else
	{
		vks::tools::exitFatal(
			"Could not open the glTF file.\n\nThe file is part of the additional "
			"asset pack.\n\nRun \"download_assets.py\" in the repository root to "
			"download the latest version.",
			-1);
		return;
	}

	// Create and upload vertex and index buffer
	// We will be using one single vertex buffer and one single index buffer for
	// the whole glTF scene Primitives (of the glTF model) will then index into
	// these using index offsets

	size_t vertexBufferSize =
		vertexBuffer.size() * sizeof(VulkanglTFScene::Vertex);

	printf("Vertex Buffer Size: %lu\n", vertexBufferSize);
	size_t indexBufferSize  = indexBuffer.size() * sizeof(uint32_t);
	glTFScene.indices.count = static_cast<uint32_t>(indexBuffer.size());

	struct StagingBuffer
	{
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertexStaging, indexStaging;

	// Create host visible staging buffers (source)
	VK_CHECK_RESULT(
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                               vertexBufferSize, &vertexStaging.buffer,
	                               &vertexStaging.memory, vertexBuffer.data()));
	// Index data
	VK_CHECK_RESULT(
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                               indexBufferSize, &indexStaging.buffer,
	                               &indexStaging.memory, indexBuffer.data()));

	// Create device local buffers (target)
	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBufferSize,
		&glTFScene.vertices.buffer, &glTFScene.vertices.memory));
	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBufferSize,
		&glTFScene.indices.buffer, &glTFScene.indices.memory));

	// Copy data from staging buffers (host) do device local buffer (gpu)
	VkCommandBuffer copyCmd =
		vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
	VkBufferCopy copyRegion = {};

	copyRegion.size = vertexBufferSize;
	vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, glTFScene.vertices.buffer, 1,
	                &copyRegion);

	copyRegion.size = indexBufferSize;
	vkCmdCopyBuffer(copyCmd, indexStaging.buffer, glTFScene.indices.buffer, 1,
	                &copyRegion);

	vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

	// Free staging resources
	vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
	vkFreeMemory(device, vertexStaging.memory, nullptr);
	vkDestroyBuffer(device, indexStaging.buffer, nullptr);
	vkFreeMemory(device, indexStaging.memory, nullptr);
}

void VulkanExample::loadAssets()
{
	loadglTFFile(getAssetPath() + "models/sponza/sponza.gltf");
}

void VulkanExample::setupDescriptors()
{
	/*
		This sample uses separate descriptor sets (and layouts) for the
     	matrices and materials (textures)
  	*/
  

	// ========================================================================
	//							SETUP FOR POOL
	// ========================================================================
	// One ubo to pass dynamic data to the shader
	// Two combined image samplers per material as each material uses color and
	// normal maps
	std::vector<VkDescriptorPoolSize> poolSizes = {
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(glTFScene.materials.size()) * 2 + 3), // +1 for multiview
	};

	// One set for matrices and one per model image/texture
	const uint32_t maxSetCount                    = static_cast<uint32_t>(glTFScene.images.size()) + 3;
	VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
	VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));


	// ========================================================================
	//							SETUP FOR MATRIX SETS
	// ========================================================================

	// Descriptor set layout for passing matrices
	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
	};

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptor_set_layouts.matrices));


	// ========================================================================
	//							SETUP FOR MATERIAL SETS
	// ========================================================================

	// Descriptor set layout for passing material textures
	setLayoutBindings = {
		// Color map
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		// Normal map
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
	};

	descriptorSetLayoutCI.pBindings    = setLayoutBindings.data();
	descriptorSetLayoutCI.bindingCount = 2;
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptor_set_layouts.textures));


	// ========================================================================
	//							MULTIVIEW PIPELINE LAYOUT
	// ========================================================================


	// Pipeline layout using both descriptor sets (set 0 = matrices, set 1 =
	// material)
	std::array<VkDescriptorSetLayout, 2> setLayouts = {descriptor_set_layouts.matrices, descriptor_set_layouts.textures};
	VkPipelineLayoutCreateInfo pipelineLayoutCI     = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

	// We will use push constants to push the local matrices of a primitive to the
	// vertex shader
	VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
	// Push constant ranges are part of the pipeline layout
	pipelineLayoutCI.pushConstantRangeCount = 1;
	pipelineLayoutCI.pPushConstantRanges    = &pushConstantRange;
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipeline_layouts.multiview));

	// Descriptor set for scene matrices
	VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptor_set_layouts.matrices, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptor_sets.multiview));

	VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptor_sets.multiview, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
	vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

	// Descriptor sets for materials
	for(uint32_t i = 0; i < glTFScene.materials.size(); i++)
	{
		const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptor_set_layouts.textures, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &glTFScene.materials[i].descriptorSet));

		VkDescriptorImageInfo colorMap  = glTFScene.getTextureDescriptor(glTFScene.materials[i].baseColorTextureIndex);
		VkDescriptorImageInfo normalMap = glTFScene.getTextureDescriptor(glTFScene.materials[i].normalTextureIndex);

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(glTFScene.materials[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &colorMap),
			vks::initializers::writeDescriptorSet(glTFScene.materials[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &normalMap),
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}


	// ========================================================================
	//							VIEWDISP SET LAYOUT
	// ========================================================================
	// descriptor set layout
	std::vector<VkDescriptorSetLayoutBinding> viewdisp_layout_bindings = {
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1),
	};

	VkDescriptorSetLayoutCreateInfo viewdisp_desc_layout = vks::initializers::descriptorSetLayoutCreateInfo(viewdisp_layout_bindings);
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &viewdisp_desc_layout, nullptr, &descriptor_set_layouts.viewdisp));


	// ========================================================================
	//							VIEWDISP PIPELINE LAYOUT
	// ========================================================================

	// pipeline layout
	VkPipelineLayoutCreateInfo viewdisp_pl_layout_ci = vks::initializers::pipelineLayoutCreateInfo(&descriptor_set_layouts.viewdisp, 1);
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &viewdisp_pl_layout_ci, nullptr, &pipeline_layouts.viewdisp));

	// Setup viewdisp descriptor set
	VkDescriptorSetAllocateInfo set_ai = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptor_set_layouts.viewdisp, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &set_ai, &descriptor_sets.viewdisp));

	// Write descsets for viewdisp pipeline
	std::vector<VkWriteDescriptorSet> viewdisp_write_desc_sets = {
		vks::initializers::writeDescriptorSet(descriptor_sets.viewdisp, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &multiview_pass.descriptor),
	};
	vkUpdateDescriptorSets(device, viewdisp_write_desc_sets.size(), viewdisp_write_desc_sets.data(), 0, nullptr);
}

void VulkanExample::preparePipelines()
{
	// ========================================================================
	//							MULTIVIEW PROPERTIES SETUP
	// ========================================================================
	VkPhysicalDeviceFeatures2KHR multiview_device_features2{};
	VkPhysicalDeviceMultiviewFeaturesKHR multiview_extension_features{};
	multiview_extension_features.sType                                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR;
	multiview_device_features2.sType                                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
	multiview_device_features2.pNext                                    = &multiview_extension_features;
	PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
	vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &multiview_device_features2);
	std::cout << "Multiview features:" << std::endl;
	std::cout << "\tmultiview = " << multiview_extension_features.multiview << std::endl;
	std::cout << "\tmultiviewGeometryShader = " << multiview_extension_features.multiviewGeometryShader << std::endl;
	std::cout << "\tmultiviewTessellationShader = " << multiview_extension_features.multiviewTessellationShader << std::endl;
	std::cout << std::endl;

	VkPhysicalDeviceProperties2KHR device_properties2{};
	VkPhysicalDeviceMultiviewPropertiesKHR extension_properties{};
	extension_properties.sType                                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR;
	device_properties2.sType                                                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
	device_properties2.pNext                                                = &extension_properties;
	PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR"));
	vkGetPhysicalDeviceProperties2KHR(physicalDevice, &device_properties2);
	std::cout << "Multiview properties:" << std::endl;
	std::cout << "\tmaxMultiviewViewCount = " << extension_properties.maxMultiviewViewCount << std::endl;
	std::cout << "\tmaxMultiviewInstanceIndex = " << extension_properties.maxMultiviewInstanceIndex << std::endl;


	// ========================================================================
	//							GENERAL GRAPHICS PIPELINE SETUP
	// ========================================================================

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
	VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
	VkPipelineColorBlendAttachmentState blendAttachmentStateCI  = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
	VkPipelineColorBlendStateCreateInfo colorBlendStateCI       = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
	VkPipelineDepthStencilStateCreateInfo depthStencilStateCI   = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
	VkPipelineViewportStateCreateInfo viewportStateCI           = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
	VkPipelineMultisampleStateCreateInfo multisampleStateCI     = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
	const std::vector<VkDynamicState> dynamicStateEnables       = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicStateCI             = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);

	std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;


	// ========================================================================
	//							MULTIVIEW GRAPHICS PIPELINE SETUP
	// ========================================================================
	const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
		vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFScene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
	};

	const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
		vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFScene::Vertex, pos)),
		vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFScene::Vertex, normal)),
		vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFScene::Vertex, uv)),
		vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFScene::Vertex, color)),
		vks::initializers::vertexInputAttributeDescription(0, 4, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFScene::Vertex, tangent)),
	};
	VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo(vertexInputBindings, vertexInputAttributes);

	VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipeline_layouts.multiview, multiview_pass.renderpass, 0);
	pipelineCI.pVertexInputState            = &vertexInputStateCI;
	pipelineCI.pInputAssemblyState          = &inputAssemblyStateCI;
	pipelineCI.pRasterizationState          = &rasterizationStateCI;
	pipelineCI.pColorBlendState             = &colorBlendStateCI;
	pipelineCI.pMultisampleState            = &multisampleStateCI;
	pipelineCI.pViewportState               = &viewportStateCI;
	pipelineCI.pDepthStencilState           = &depthStencilStateCI;
	pipelineCI.pDynamicState                = &dynamicStateCI;
	pipelineCI.stageCount                   = static_cast<uint32_t>(shaderStages.size());
	pipelineCI.pStages                      = shaderStages.data();


	shaderStages[0] = loadShader(getShadersPath() + "gltfscenerendering/multiview.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
	shaderStages[1] = loadShader(getShadersPath() + "gltfscenerendering/multiview.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);


	// POI: Instead if using a few fixed pipelines, we create one pipeline for
	// each material using the properties of that material
	for(VulkanglTFScene::Material &material : glTFScene.materials)
	{
		struct MaterialSpecializationData
		{
			VkBool32 alphaMask;
			float alphaMaskCutoff;
		} materialSpecializationData;

		materialSpecializationData.alphaMask       = material.alphaMode == "MASK";
		materialSpecializationData.alphaMaskCutoff = material.alphaCutOff;

		// POI: Constant fragment shader material parameters will be set using
		// specialization constants
		std::vector<VkSpecializationMapEntry> specializationMapEntries = {
			vks::initializers::specializationMapEntry(0, offsetof(MaterialSpecializationData, alphaMask), sizeof(MaterialSpecializationData::alphaMask)),
			vks::initializers::specializationMapEntry(1, offsetof(MaterialSpecializationData, alphaMaskCutoff), sizeof(MaterialSpecializationData::alphaMaskCutoff)),
		};

		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(specializationMapEntries, sizeof(materialSpecializationData), &materialSpecializationData);
		shaderStages[1].pSpecializationInfo     = &specializationInfo;

		// For double sided materials, culling will be disabled
		rasterizationStateCI.cullMode = material.doubleSided ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &material.pipeline));
	}


	// ========================================================================
	//							VIEWDISP GRAPHICS PIPELINE SETUP
	// ========================================================================

	// Set the rasterization state cullmode to frontbit
	rasterizationStateCI.cullMode = VK_CULL_MODE_FRONT_BIT;

	// Viewdisp pipelines setup
	// Also make the semaphore
	VkSemaphoreCreateInfo semaphore_ci = vks::initializers::semaphoreCreateInfo();
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphore_ci, nullptr, &multiview_pass.semaphore));

	// Viewdisplay for multiview
	VkPipelineShaderStageCreateInfo viewdisp_shader_stages[2];
	float multiview_array_layer                                = 0.0f;
	VkSpecializationMapEntry viewdisp_specialization_map_entry = {0, 0, sizeof(float)};
	VkSpecializationInfo viewdisp_specialization_info          = {
        .mapEntryCount = 1,
        .pMapEntries   = &viewdisp_specialization_map_entry,
        .dataSize      = sizeof(float),
        .pData         = &multiview_array_layer,
    };


	for(uint32_t i = 0; i < 2; i++)
	{
		viewdisp_shader_stages[0]                     = loadShader(getShadersPath() + "gltfscenerendering/viewdisplay.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		viewdisp_shader_stages[1]                     = loadShader(getShadersPath() + "gltfscenerendering/viewdisplay.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		viewdisp_shader_stages[1].pSpecializationInfo = &viewdisp_specialization_info;
		multiview_array_layer                         = (float) (1 - i);

		VkPipelineVertexInputStateCreateInfo empty_input_state = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCI.pVertexInputState                           = &empty_input_state;
		pipelineCI.layout                                      = pipeline_layouts.viewdisp;
		pipelineCI.pStages                                     = viewdisp_shader_stages;
		pipelineCI.renderPass                                  = renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &viewdisp_pipelines[i]));
	}
}

void VulkanExample::prepareUniformBuffers()
{
	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&shaderData.buffer, sizeof(shaderData.values)));

	VK_CHECK_RESULT(shaderData.buffer.map());
	updateUniformBuffers();
}

void VulkanExample::updateUniformBuffers()
{
	// Matrices for the two viewports
	// See http://paulbourke.net/stereographics/stereorender/

	// Calculate some variables
	float aspectRatio = (float) (width * 0.5f) / (float) height;
	float wd2         = zNear * tan(glm::radians(fov / 2.0f));
	float ndfl        = zNear / focalLength;
	float left, right;
	float top    = wd2;
	float bottom = -wd2;

	glm::vec3 camFront;
	camFront.x         = -cos(glm::radians(camera.rotation.x)) * sin(glm::radians(camera.rotation.y));
	camFront.y         = -sin(glm::radians(camera.rotation.x));
	camFront.z         = cos(glm::radians(camera.rotation.x)) * cos(glm::radians(camera.rotation.y));
	camFront           = glm::normalize(camFront);
	glm::vec3 camRight = glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f)));

	glm::mat4 rotM = glm::mat4(1.0f);
	glm::mat4 transM;

	rotM = glm::rotate(rotM, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	rotM = glm::rotate(rotM, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	rotM = glm::rotate(rotM, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

	for(int i = 0; i < 4; i++)
	{
		rotM[i][0] *= -1.0f;
	}

	// Left eye
	left  = -aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;
	right = aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;

	transM = glm::translate(glm::mat4(1.0f), camera.position - camRight * (eyeSeparation / 2.0f));

	shaderData.values.projection[0] = glm::frustum(left, right, bottom, top, zNear, zFar);
	shaderData.values.view[0]       = rotM * transM;

	// Right eye
	left  = -aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;
	right = aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;

	transM = glm::translate(glm::mat4(1.0f), camera.position + camRight * (eyeSeparation / 2.0f));

	shaderData.values.projection[1] = glm::frustum(left, right, bottom, top, zNear, zFar);
	shaderData.values.view[1]       = rotM * transM;

	memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));
}


void VulkanExample::prepare()
{
	VulkanExampleBase::prepare();
	//setup_opencl();
	loadAssets();
	setup_multiview();
	prepareUniformBuffers();
	setupDescriptors();
	preparePipelines();
	foveal_regions = create_image_packet();
	server         = Server();
	server.connect_to_client(PORT);
	buildCommandBuffers();

	VkFenceCreateInfo multiview_fence_ci = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
	multiview_pass.wait_fences.resize(multiview_pass.command_buffers.size());
	for(uint32_t i = 0; i < multiview_pass.wait_fences.size(); i++)
	{
		VK_CHECK_RESULT(vkCreateFence(device, &multiview_fence_ci, nullptr, &multiview_pass.wait_fences[i]));
	}

	setup_video_encoder();
	setup_video_decoder();


	prepared = true;
}


// This will only encode one frame at a time
static void encode(VulkanExample *ve, AVCodecContext *encode_context, AVFrame *frame, AVPacket *packet, FILE *outfile)
{
	timeval encode_end_time;

	int ret = avcodec_send_frame(encode_context, frame);
	if(ret < 0)
	{
		char errbuf[64];
		int err = av_strerror(ret, errbuf, 64);

		printf("Error: %s\n", &errbuf[0]);
		//av_make_error_string(errbuf, 64, ret);

		printf("Ret: %d\n", ret);
		throw std::runtime_error("Error sending frame for encoding");
	}

	ret = avcodec_receive_packet(encode_context, packet);

	gettimeofday(&encode_end_time, nullptr);
	float encode_time_diff = vku::time_difference(ve->tmp_timers.encode_start_time, encode_end_time);
	ve->timers.encode_time.push_back(encode_time_diff);

	if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
	{
		// dbg section
		// Send garbage data twice to advance shit
		uint32_t garbage = 4;
		int sendret = send(ve->server.client_fd[0], &garbage, sizeof(uint32_t), 0);
		send(ve->server.client_fd[0], &garbage, sizeof(garbage), 0);			
	}

	else if(ret < 0)
	{
		throw std::runtime_error("Could not receive packet");
	}

	else
	{
		send(ve->server.client_fd[0], &packet->size, sizeof(packet->size), 0);
		ssize_t sendret = send(ve->server.client_fd[0], &packet->data[0], packet->size, 0);
		ve->should_wait_for_camera_data = true;
	}
}


void VulkanExample::setup_video_encoder()
{ 
	const char *filename, *codec_name;
	int ret;

	filename   = "file.mp4";
	codec_name = "h264_nvenc";

	/* find the mpeg1video encoder */
	encoder.codec = avcodec_find_encoder_by_name(codec_name);
	if(!encoder.codec)
	{
		fprintf(stderr, "Codec '%s' not found\n", codec_name);
		exit(1);
	}

	encoder.c = avcodec_alloc_context3(encoder.codec);
	if(!encoder.c)
	{
		fprintf(stderr, "Could not allocate video codec context\n");
		exit(1);
	}

	encoder.packet = av_packet_alloc();
	encoder.frame = av_frame_alloc();
}


struct EncodingData
{
	VulkanExample *ve;
	uint8_t *y; // not going to b e used while testing ffmpeg colour space conversion
	uint8_t *u; // not going to b e used while testing ffmpeg colour space conversion
	uint8_t *v; // not going to b e used while testing ffmpeg colour space conversion
};


static void *receive_camera_data(void *host_renderer)
{
	VulkanExample *ve = (VulkanExample*) host_renderer;

	float camera_buf[6];

	int client_read = recv(ve->server.client_fd[1], camera_buf, 6 * sizeof(float), MSG_WAITALL);
	ve->camera.position = glm::vec3(camera_buf[0], camera_buf[1], camera_buf[2]);
	ve->camera.rotation = glm::vec3(camera_buf[3], camera_buf[4], camera_buf[5]);

	return nullptr;
}


static void *begin_video_encoding(void *void_encoding_data) // uint8_t *luminance_y, uint8_t *bp_u, uint8_t *rp_v)
{
	timeval encode_end_time;
	gettimeofday(&ve->tmp_timers.encode_start_time, nullptr);
	VulkanExample *ve = (VulkanExample*) void_encoding_data;

	int i;
	uint8_t endcode[] = {0, 0, 1, 0xb7};
	FILE *f;
	std::string filename = "h264encoding" + std::to_string(ve->numframes) + ".mp4";


	//ve->encoder.packet = av_packet_alloc();
	if(!ve->encoder.packet)
		exit(1);

	/* put sample parameters */
	//c->bit_rate = 400000;
	/* resolution must be a multiple of two */
	ve->encoder.c->width  = 2 * FOVEAWIDTH;
	ve->encoder.c->height = FOVEAHEIGHT;
	/* frames per second */
	ve->encoder.c->time_base = (AVRational){1, 60};
	// c->framerate = (AVRational){25, 1};

	/* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
	// c->gop_size		= 10;
	// c->max_b_frames = 1;
	ve->encoder.c->pix_fmt		= AV_PIX_FMT_YUV444P;
	//ve->encoder.

	//if(ve->encoder.codec->id == AV_CODEC_ID_H264)
	//	av_opt_set(ve->encoder.c->priv_data, "preset", "slow", 0);
	av_opt_set(ve->encoder.c->priv_data, "crf", "1", AV_OPT_SEARCH_CHILDREN);
	av_opt_set(ve->encoder.c->priv_data, "qp", "1", AV_OPT_SEARCH_CHILDREN);

	/* open it */
	int ret = avcodec_open2(ve->encoder.c,ve-> encoder.codec, NULL);
	if(ret < 0)
	{
		throw std::runtime_error("Could not open codec!");
	}

	//ve->encoder.frame = av_frame_alloc();
	ve->encoder.frame->format = AV_PIX_FMT_YUV444P;
	ve->encoder.frame->width = 2 * FOVEAWIDTH;
	ve->encoder.frame->height = FOVEAHEIGHT;
	ve->encoder.frame->pict_type = AV_PICTURE_TYPE_I;
	av_frame_get_buffer(ve->encoder.frame, 1);


	/* Make sure the frame data is writable.
		On the first round, the frame is fresh from av_frame_get_buffer()
		and therefore we know it is writable.
		But on the next rounds, encode() will have called
		avcodec_send_frame(), and the codec may have kept a reference to
		the frame in its internal structures, that makes the frame
		unwritable.
		av_frame_make_writable() checks that and allocates a new buffer
		for the frame only if necessary.
		*/
	ret = av_frame_get_buffer(ve->encoder.frame, 0);

	// Get the context or whatever
	SwsContext *sws_ctx = sws_getContext(ve->encoder.c->width, ve->encoder.c->height, AV_PIX_FMT_RGBA, ve->encoder.c->width, ve->encoder.c->height, AV_PIX_FMT_YUV444P, 0, 0, 0, 0);

	fflush(stdout);

	ret = av_frame_make_writable(ve->encoder.frame);
	if(ret < 0)
	{
		throw std::runtime_error("Could not make av frame writeable");
	}

	vkMapMemory(ve->device, ve->foveal_regions.buffer.memory, 0, 2 * FOVEAWIDTH * FOVEAHEIGHT * sizeof(uint32_t), 0, (void**) &ve->foveal_regions.data);

	int in_line_size[1] = {2 * ve->encoder.c->width};
	uint8_t *in_data[1] = {(uint8_t*) ve->foveal_regions.data};
	ve->encoder.frame->pts = 0;
	sws_scale(sws_ctx, in_data, in_line_size, 0, ve->encoder.c->height, ve->encoder.frame->data, ve->encoder.frame->linesize);

	encode(ve, ve->encoder.c, ve->encoder.frame, ve->encoder.packet, f);
	
	vkUnmapMemory(ve->device, ve->foveal_regions.buffer.memory);

	// finish encoding timing inside encode()

	//av_frame_free(&ve->encoder.frame);

	return nullptr;
}


void VulkanExample::setup_video_decoder()
{
	// not sure if it's necessary to set end of buffer to 0
	decoder.codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	if(!decoder.codec)
	{
		throw std::runtime_error("Decoder: Could not find H264 encoder");
	}

	decoder.parser = av_parser_init(decoder.codec->id);
	if(!decoder.parser)
	{
		throw std::runtime_error("Decoder: Could not find parser");
	}
	
}

static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize, char *filename)
{
    FILE *f;
    int i;
 
    f = fopen(filename,"wb");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
	{
        fwrite(buf + i * wrap, 1, xsize, f);
	}
	fclose(f);
}


void VulkanExample::decode(AVCodecContext *decode_context, AVFrame *frame, AVPacket *packet, const char *filename)
{
	printf("%d Decode called\n", numframes);
	char buf[1024];
	int ret = avcodec_send_packet(decode_context, packet);
	if(ret < 0)
	{
		throw std::runtime_error("Decode: Error sending a packet");
	}
	
	while(ret >= 0)
	{
		printf("%d In loop of decode\n", numframes);
		ret = avcodec_receive_frame(decode_context, frame);
		printf("%d AV Frame Received\n", numframes);
		if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
		{
			printf("AV Error on ret after receiving frame; not writing output\n");
			return;
		}
		else if(ret < 0)
		{
			throw std::runtime_error("Error during decoding");
		}
	
		printf("saving frame %3d\n", decode_context->frame_number);
		fflush(stdout);
	
		/* the picture is allocated by the decoder. no need to
			free it */
		snprintf(buf, sizeof(buf), "%s-%d", filename, decode_context->frame_number);
		pgm_save(frame->data[0], frame->linesize[0],
					frame->width, frame->height, buf);
	}
}


void VulkanExample::begin_video_decoding()
{
	decoder.c = avcodec_alloc_context3(decoder.codec);
	if(!decoder.c)
	{
		throw std::runtime_error("Decoder: Could not allocate video codec context");
	}

	// Open the codec
	if(avcodec_open2(decoder.c, decoder.codec, nullptr) < 0)
	{
		throw std::runtime_error("Decoder: Could not open codec");
	}

	decoder.packet = av_packet_alloc();
	if(!decoder.packet)
	{
		throw std::runtime_error("Decoder: Could not alloc packet!");
	}
	

	int INBUF_SIZE = FOVEAWIDTH * FOVEAHEIGHT * 3;
	uint8_t *data;
	uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
	std::string filename = "h264encoding" + std::to_string(numframes) + ".mp4";
	std::string outfilename = "pgmh264encoding" + std::to_string(numframes) + ".pgm";
	FILE *f = fopen(filename.c_str(), "rb");
	if(!f)
	{
		throw std::runtime_error("Decoder: Could not open file");
	}

	decoder.frame = av_frame_alloc();
	if(!decoder.frame)
	{
		throw std::runtime_error("Decoder: Could not allocate video frame");
	}

	int eof;
	do
	{
		int data_size = fread(inbuf, 1, INBUF_SIZE, f);
		printf("DATA SIZE: %d\n", data_size);
		if(ferror(f))
		{
			printf("Decoder (not crashing yet): ferror on reading file data\n");
			break;
		}

		eof = !data_size;
		data = inbuf;
		while(data_size > 0 || eof)
		{
			printf("%d In loop of begin_decoder\n", numframes);
			int ret = av_parser_parse2(decoder.parser, decoder.c, &decoder.packet->data, &decoder.packet->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
			printf("RET: %d\n", ret);
			if(ret < 0)
			{
				throw std::runtime_error("Decoder: Error while parsing");
			}

			data += ret;
			data_size -= ret;

			if(decoder.packet->size)
			{
				decode(decoder.c, decoder.frame, decoder.packet, outfilename.c_str());
			}

			else if(eof)
			{
				break;
			}
		}
	} while (!eof);

	// flush decoder

	decode(decoder.c, decoder.frame, nullptr, outfilename.c_str());

	fclose(f);


	avcodec_free_context(&decoder.c);
	av_frame_free(&decoder.frame);
	av_packet_free(&decoder.packet);
}


/*void VulkanExample::rgba_to_rgb_opencl(const uint8_t *__restrict__ in_h, uint8_t *__restrict__ out_Y_h, uint8_t *__restrict__ out_U_h, uint8_t *__restrict__ out_V_h, size_t in_len)
{
	size_t out_len = in_len / 4;	
	cl::Buffer in_d(cl.context, CL_MEM_READ_ONLY, sizeof(uint8_t) * in_len);
	cl::Buffer out_Y_d(cl.context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * out_len);
	cl::Buffer out_U_d(cl.context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * out_len);
	cl::Buffer out_V_d(cl.context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * out_len);

	cl.queue.enqueueWriteBuffer(in_d, CL_TRUE, 0, sizeof(uint8_t) * in_len, in_h);

	cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> cl_rgba_to_rgb(cl::Kernel(cl.alpha_removal_program, "cl_rgba_to_rgb"));
	cl::NDRange global(in_len);
	cl_rgba_to_rgb(cl::EnqueueArgs(cl.queue, global), in_d, out_Y_d, out_U_d, out_V_d).wait();

	cl.queue.enqueueReadBuffer(out_Y_d, CL_TRUE, 0, sizeof(uint8_t) * out_len, out_Y_h);
	cl.queue.enqueueReadBuffer(out_U_d, CL_TRUE, 0, sizeof(uint8_t) * out_len, out_U_h);
	cl.queue.enqueueReadBuffer(out_V_d, CL_TRUE, 0, sizeof(uint8_t) * out_len, out_V_h);
}*/


/*void VulkanExample::setup_opencl()
{
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if(all_platforms.size() == 0)
	{
		throw std::runtime_error("No OpenCL platforms found");
	}

 	cl.platform = all_platforms[0];
	printf("Using platform: %s\n", cl.platform.getInfo<CL_PLATFORM_NAME>().c_str());

	std::vector<cl::Device> all_devices;
	cl.platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	cl.device = all_devices[0];
	printf("Using OpenCL device: %s\n", cl.device.getInfo<CL_DEVICE_NAME>().c_str());

	cl::Context ctx({cl.device});
	cl.context = ctx;

	cl::CommandQueue cmdqueue(cl.context, cl.device);
	cl.queue = cmdqueue;

	// Load kernel
	std::string rgba_to_rgb_kernel_str_code = 
		"kernel void cl_rgba_to_rgb(global const char *in, global char *out_Y, global char *out_U, global char *out_V)"
		"{"
		"	int in_idx = get_global_id(0);"

		"	if(in_idx % 4 == 0)"
		"	{"
		"		out_Y[in_idx / 4] = in[in_idx];"
		"	}"

		"	else if(in_idx % 4 == 1)"
		"	{"
		"		out_U[in_idx / 4] = in[in_idx];"
		"	}"

		"	else if(in_idx % 4 == 2)"
		"	{"
		"		out_V[in_idx / 4] = in[in_idx];"
		"	}"

		"	barrier(CLK_GLOBAL_MEM_FENCE);"
		"}";

	cl.sources.push_back({rgba_to_rgb_kernel_str_code.c_str(), rgba_to_rgb_kernel_str_code.length()});
	cl::Program program(cl.context, cl.sources);
	cl.alpha_removal_program = program;

	if(cl.alpha_removal_program.build({cl.device}) != CL_SUCCESS)
	{
		std::cout << "Error building: " << cl.alpha_removal_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl.device) << "\n";
		exit(-1);
	}


	// Tests for kernel
	char *in_h = new char[16];
	char *out_Y_h = new char[4];
	char *out_U_h = new char[4];
	char *out_V_h = new char[4];

	for(uint32_t i = 0; i < 16; i++)
	{
		if(i % 4 == 0)
		{
			in_h[i] = (char) 1;
		}

		else if(i % 4 == 1)
		{
			in_h[i] = (char) 2;
		}
		
		else if(i % 4 == 2)
		{
			in_h[i] = (char) 3;
		}

		else
		{
			in_h[i] = (char) 4;
		}
	}
	

	for(uint32_t i = 0; i < 16; i++)
	{
		printf("%d in_h: %d\n", i, in_h[i]);
	}

	cl::Buffer in_d(cl.context, CL_MEM_READ_ONLY, sizeof(char) * 16);
	cl::Buffer out_Y_d(cl.context, CL_MEM_WRITE_ONLY, sizeof(char) * 4);
	cl::Buffer out_U_d(cl.context, CL_MEM_WRITE_ONLY, sizeof(char) * 4);
	cl::Buffer out_V_d(cl.context, CL_MEM_WRITE_ONLY, sizeof(char) * 4);

	cl.queue.enqueueWriteBuffer(in_d, CL_TRUE, 0, sizeof(char) * 16, in_h);

	cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> cl_rgba_to_rgb(cl::Kernel(cl.alpha_removal_program, "cl_rgba_to_rgb"));
	cl::NDRange global(16);
	cl_rgba_to_rgb(cl::EnqueueArgs(cl.queue, global), in_d, out_Y_d, out_U_d, out_V_d).wait();

	cl.queue.enqueueReadBuffer(out_Y_d, CL_TRUE, 0, sizeof(char) * 4, out_Y_h);
	cl.queue.enqueueReadBuffer(out_U_d, CL_TRUE, 0, sizeof(char) * 4, out_U_h);
	cl.queue.enqueueReadBuffer(out_V_d, CL_TRUE, 0, sizeof(char) * 4, out_V_h);

	for(uint32_t i = 0; i < 4; i++)
	{
		printf("%d out_Y_h: %d\n", i, out_Y_h[i]);
	}

	for(uint32_t i = 0; i < 4; i++)
	{
		printf("%d out_U_h: %d\n", i, out_U_h[i]);
	}

	for(uint32_t i = 0; i < 4; i++)
	{
		printf("%d out_V_h: %d\n", i, out_V_h[i]);
	}
}*/


void VulkanExample::draw()
{
	printf("\n");
	printf("Framenum: %d\n", numframes);
	timeval drawstarttime;
	timeval drawendtime;
	gettimeofday(&drawstarttime, nullptr);

	VulkanExampleBase::prepareFrame();
	should_wait_for_camera_data = false;

	// Multiview offscreen render
	VK_CHECK_RESULT(vkWaitForFences(device, 1, &multiview_pass.wait_fences[currentBuffer], VK_TRUE, UINT64_MAX));
	VK_CHECK_RESULT(vkResetFences(device, 1, &multiview_pass.wait_fences[currentBuffer]));
	submitInfo.pWaitSemaphores    = &semaphores.presentComplete;
	submitInfo.pSignalSemaphores  = &multiview_pass.semaphore;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers    = &multiview_pass.command_buffers[currentBuffer];
	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, multiview_pass.wait_fences[currentBuffer]));


	// View display
	VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
	VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
	submitInfo.pWaitSemaphores    = &multiview_pass.semaphore;
	submitInfo.pSignalSemaphores  = &semaphores.renderComplete;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers    = &drawCmdBuffers[currentBuffer];
	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]));

	VkSwapchainKHR swapchains_to_present_to[] = {swapChain.swapChain};
	VkPresentInfoKHR present_info             = {
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &semaphores.renderComplete,
        .swapchainCount     = 1,
        .pSwapchains        = swapchains_to_present_to,
        .pImageIndices      = &currentBuffer,
    };


	VulkanExampleBase::submitFrame();

	// Determine area to copy
	int32_t midpoint_of_eye_x = SERVERWIDTH / 4;
	int32_t midpoint_of_eye_y = SERVERHEIGHT / 2;

	// Get the top left point for left eye
	int32_t topleft_lefteye_x  = midpoint_of_eye_x - (FOVEAWIDTH / 2);
	int32_t topleft_eyepoint_y = midpoint_of_eye_y - (FOVEAHEIGHT / 2);

	// Get the top left point for right eye -- y is same
	int32_t topleft_righteye_x = (SERVERWIDTH / 2) + midpoint_of_eye_x - (FOVEAWIDTH / 2);

	VkOffset3D lefteye_copy_offset = {
		.x = topleft_lefteye_x,
		.y = topleft_eyepoint_y,
		.z = 0,
	};

	VkOffset3D righteye_copy_offset = {
		.x = topleft_righteye_x,
		.y = topleft_eyepoint_y,
		.z = 0,
	};

	gettimeofday(&drawendtime, nullptr);
	timers.drawtime.push_back(vku::time_difference(drawstarttime, drawendtime));

	timeval copystarttime;
	timeval copyendtime;
	gettimeofday(&copystarttime, nullptr);

	// Now copy the image packet back
	foveal_regions = copy_image_to_packet(swapChain.images[currentBuffer], foveal_regions, lefteye_copy_offset, righteye_copy_offset);

	gettimeofday(&copyendtime, nullptr);
	timers.copy_image_time.push_back(vku::time_difference(copystarttime, copyendtime));

	size_t input_framesize_bytes  = FOVEAWIDTH * FOVEAHEIGHT * sizeof(uint32_t);
	uint8_t left_out_Y_h[input_framesize_bytes / 4];
	uint8_t left_out_U_h[input_framesize_bytes / 4];
	uint8_t left_out_V_h[input_framesize_bytes / 4];
	//rgba_to_rgb_opencl((uint8_t*) lefteye_fovea.data, left_out_Y_h, left_out_U_h, left_out_V_h, input_framesize_bytes);

	int receive_camera_thread = pthread_create(&vk_pthread.recv_camera, nullptr, receive_camera_data, this);
	int left_image_send_encode = pthread_create(&vk_pthread.send_image, nullptr, begin_video_encoding, this);
	pthread_join(vk_pthread.send_image, nullptr);
	pthread_join(vk_pthread.recv_camera, nullptr);


	if(timers.drawtime.size() == 1024)
	{
		int len = 1024 * sizeof(float) * 6;
		float databuf[len];
		int server_read = recv(server.client_fd[0], databuf, len, MSG_WAITALL);

		std::string filename = "CLIENTDATA.tsv";
		std::ofstream file(filename, std::ios::out | std::ios::binary);
		file << "recvswapchain\tsendcamera\tdecode\tcopyintoswap\tnetframetime\tmbps\n";

		for(uint32_t i = 1; i < 1000; i++)
		{
			std::string datapointstr = std::to_string(databuf[i]) + "\t" +
			                           std::to_string(databuf[i + 1024 * 1]) + "\t" +
			                           std::to_string(databuf[i + 1024 * 2]) + "\t" +
			                           std::to_string(databuf[i + 1024 * 3]) + "\t" +
			                           std::to_string(databuf[i + 1024 * 4]) + "\t" +
			                           std::to_string(databuf[i + 1024 * 5]) + "\n";
			file << datapointstr;
		}

		file.close();

		// Write server data
		filename = "SERVERDATA.tsv";
		std::ofstream file2(filename, std::ios::out | std::ios::binary);
		file2 << "drawtime\tencode\tcopytime\n";

		for(uint32_t i = 1; i < 1000; i++)
		{
			std::string datapointstr = std::to_string(timers.drawtime[i]) + "\t" +
			                           std::to_string(timers.encode_time[i]) + "\t" +
			                           std::to_string(timers.copy_image_time[i]) + "\n";

			file2 << datapointstr;
		}

		file2.close();
	}

	numframes++;
}


ImagePacket VulkanExample::copy_image_to_packet(VkImage src_image, ImagePacket image_packet, VkOffset3D left_offset, VkOffset3D right_offset)
{
	ImagePacket dst                = image_packet;
	VkCommandBuffer copy_cmdbuffer = vku::begin_command_buffer(device, cmdPool);

	//printf("Command buffer begun\n");

	// Transition swapchain image from present to source transfer layout
	vku::transition_image_layout(device, cmdPool, copy_cmdbuffer,
	                             src_image,
	                             VK_ACCESS_MEMORY_READ_BIT,
	                             VK_ACCESS_TRANSFER_READ_BIT,
	                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                             VK_PIPELINE_STAGE_TRANSFER_BIT,
	                             VK_PIPELINE_STAGE_TRANSFER_BIT);

	//printf("Swapchain transitioned to read only VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL\n");


	VkImageSubresourceLayers image_subresource = {
		.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseArrayLayer = 0,
		.layerCount     = 1,
	};

	// Midpoint areas....
	int32_t midpoint_of_eye_x = SERVERWIDTH / 4;
	int32_t midpoint_of_eye_y = SERVERHEIGHT / 2;

	// Get the top left point for left eye
	int32_t topleft_lefteye_x  = midpoint_of_eye_x - (FOVEAWIDTH / 2);
	int32_t topleft_eyepoint_y = midpoint_of_eye_y - (FOVEAHEIGHT / 2);

	// Get the top left point for right eye -- y is same
	int32_t topleft_righteye_x = (SERVERWIDTH / 2) + midpoint_of_eye_x - (SERVERHEIGHT / 2);

	// Image offsets
	VkOffset3D lefteye_image_offset = {
		.x = topleft_lefteye_x,
		.y = topleft_eyepoint_y,
		.z = 0,
	};

	VkOffset3D righteye_image_offset = {
		.x = topleft_righteye_x,
		.y = topleft_eyepoint_y,
		.z = 0,
	};

	// Create the vkbufferimagecopy pregions
	VkBufferImageCopy left_copy_region = {
		.bufferOffset      = 0,
		.bufferRowLength   = FOVEAWIDTH,
		.bufferImageHeight = FOVEAHEIGHT,
		.imageSubresource  = image_subresource,
		.imageOffset       = lefteye_image_offset,
		.imageExtent       = {FOVEAWIDTH, FOVEAHEIGHT, 1},
	};

	VkBufferImageCopy right_copy_region = {
		.bufferOffset      = FOVEAWIDTH * FOVEAHEIGHT * sizeof(uint32_t),
		.bufferRowLength   = FOVEAWIDTH,
		.bufferImageHeight = FOVEAHEIGHT,
		.imageSubresource  = image_subresource,
		.imageOffset       = righteye_image_offset,
		.imageExtent       = {FOVEAWIDTH, FOVEAHEIGHT, 1},
	};

	vkCmdCopyImageToBuffer(copy_cmdbuffer, 
		src_image, 
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
		foveal_regions.buffer.buffer, 
		1, 
		&left_copy_region);
	
	vkCmdCopyImageToBuffer(copy_cmdbuffer, 
		src_image, 
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
		foveal_regions.buffer.buffer, 
		1, 
		&right_copy_region);

	// transition swapchain image back now that copying is done
	vku::transition_image_layout(device, cmdPool, copy_cmdbuffer,
	                             src_image,
	                             VK_ACCESS_TRANSFER_READ_BIT,
	                             VK_ACCESS_MEMORY_READ_BIT,
	                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	                             VK_PIPELINE_STAGE_TRANSFER_BIT,
	                             VK_PIPELINE_STAGE_TRANSFER_BIT);

	vku::end_command_buffer(device, queue, cmdPool, copy_cmdbuffer);
	
	return dst;
}

ImagePacket VulkanExample::create_image_packet()
{
	ImagePacket dst;

	VkDeviceSize image_buffer_size = FOVEAWIDTH * 2 * FOVEAHEIGHT * sizeof(uint32_t);
	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&dst.buffer, image_buffer_size));

	dst.num_bytes = (size_t) image_buffer_size;


	return dst;
}

void VulkanExample::write_imagepacket_to_file(ImagePacket packet, uint32_t buffer, std::string name)
{
	/*std::string filename = "tmpserver_" + name + " " + std::to_string(currentBuffer) + ".ppm";
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	file << "P6\n"
		 << FOVEAWIDTH << "\n"
		 << FOVEAHEIGHT << "\n"
		 << 255 << "\n";

	for(uint32_t y = 0; y < FOVEAHEIGHT; y++)
	{
		uint32_t *row = (uint32_t *) packet.data;
		for(uint32_t x = 0; x < FOVEAWIDTH; x++)
		{
			file.write((char *) row, 3);
			row++;
		}
		packet.data += packet.subresource_layout.rowPitch;
	}

	file.close();*/
}


void VulkanExample::render()
{
	draw();

	//if(camera.updated)
	//{
	updateUniformBuffers();
	//}
}

void VulkanExample::OnUpdateUIOverlay(vks::UIOverlay *overlay)
{
	if(overlay->header("Visibility"))
	{

		if(overlay->button("All"))
		{
			std::for_each(glTFScene.nodes.begin(), glTFScene.nodes.end(),
			              [](VulkanglTFScene::Node &node)
			              { node.visible = true; });
			buildCommandBuffers();
		}
		ImGui::SameLine();
		if(overlay->button("None"))
		{
			std::for_each(glTFScene.nodes.begin(), glTFScene.nodes.end(),
			              [](VulkanglTFScene::Node &node)
			              { node.visible = false; });
			buildCommandBuffers();
		}
		ImGui::NewLine();

		// POI: Create a list of glTF nodes for visibility toggle
		ImGui::BeginChild("#nodelist", ImVec2(200.0f, 340.0f), false);
		for(auto &node : glTFScene.nodes)
		{
			if(overlay->checkBox(node.name.c_str(), &node.visible))
			{
				buildCommandBuffers();
			}
		}
		ImGui::EndChild();
	}
}

VULKAN_EXAMPLE_MAIN()