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

#include "gltfscenerendering.h"

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
		materials[i].alphaMode	 = glTFMaterial.alphaMode;
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
			uint32_t firstIndex						 = static_cast<uint32_t>(indexBuffer.size());
			uint32_t vertexStart					 = static_cast<uint32_t>(vertexBuffer.size());
			uint32_t indexCount						 = 0;
			// Vertices
			{
				const float *positionBuffer	 = nullptr;
				const float *normalsBuffer	 = nullptr;
				const float *texCoordsBuffer = nullptr;
				const float *tangentsBuffer	 = nullptr;
				size_t vertexCount			 = 0;

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
				// Get buffer data for vertex normals
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
					vert.pos	= glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
					vert.normal = glm::normalize(
						glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
					vert.uv		 = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
					vert.color	 = glm::vec3(1.0f);
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
			primitive.firstIndex	= firstIndex;
			primitive.indexCount	= indexCount;
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
		glm::mat4 nodeMatrix				 = node.matrix;
		VulkanglTFScene::Node *currentParent = node.parent;
		while(currentParent)
		{
			nodeMatrix	  = currentParent->matrix * nodeMatrix;
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
	title		 = "glTF scene rendering";
	camera.type	 = Camera::CameraType::firstperson;
	camera.flipY = true;
	camera.setPosition(glm::vec3(0.0f, 1.0f, 0.0f));
	camera.setRotation(glm::vec3(0.0f, -90.0f, 0.0f));
	camera.setPerspective(60.0f, (float) width / (float) height, 0.1f, 256.0f);

	// Multiview setup
	// Enable extension required for multiview
	enabledDeviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);

	// Reading device properties and features for multiview requires VK_KHR_get_physical_device_properties2 to be enabled
	enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

	// Enable required extension features
	physical_device_multiview_features = {
		.sType	   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR,
		.multiview = VK_TRUE,
	};
	deviceCreatepNextChain = &physical_device_multiview_features;
}

VulkanExample::~VulkanExample()
{
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.textures, nullptr);
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
			.sType		 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext		 = nullptr,
			.flags		 = 0,
			.imageType	 = VK_IMAGE_TYPE_2D,
			.format		 = swapChain.colorFormat,
			.extent		 = {width, height, 1},
			.mipLevels	 = 1,
			.arrayLayers = multiview_layers,
			.samples	 = VK_SAMPLE_COUNT_1_BIT,
			.tiling		 = VK_IMAGE_TILING_OPTIMAL,
			.usage		 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		};
		VK_CHECK_RESULT(vkCreateImage(device, &image_ci, nullptr, &multiview_pass.colour.image));

		VkMemoryRequirements memory_requirements;
		vkGetImageMemoryRequirements(device, multiview_pass.colour.image, &memory_requirements);

		VkMemoryAllocateInfo memory_ai = {
			.sType			 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize	 = memory_requirements.size,
			.memoryTypeIndex = vulkanDevice->getMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
		};
		VK_CHECK_RESULT(vkAllocateMemory(device, &memory_ai, nullptr, &multiview_pass.colour.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, multiview_pass.colour.image, multiview_pass.colour.memory, 0));


		VkImageSubresourceRange colour_subresource = {
			.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel	= 0,
			.levelCount		= 1,
			.baseArrayLayer = 0,
			.layerCount		= multiview_layers,
		};

		VkImageViewCreateInfo image_view_ci = {
			.sType			  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext			  = nullptr,
			.flags			  = 0,
			.image			  = multiview_pass.colour.image,
			.viewType		  = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			.format			  = swapChain.colorFormat,
			.subresourceRange = colour_subresource,
		};
		VK_CHECK_RESULT(vkCreateImageView(device, &image_view_ci, nullptr, &multiview_pass.colour.view));

		VkSamplerCreateInfo sampler_ci = {
			.sType			  = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.pNext			  = nullptr,
			.flags			  = 0,
			.magFilter		  = VK_FILTER_NEAREST,
			.minFilter		  = VK_FILTER_NEAREST,
			.mipmapMode		  = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU	  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV	  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW	  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias		  = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy	  = 1.0f,
			.minLod			  = 0.0f,
			.maxLod			  = 1.0f,
			.borderColor	  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		};
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler_ci, nullptr, &multiview_pass.sampler));

		// Setup the descriptors for the colour attachment
		multiview_pass.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		multiview_pass.descriptor.imageView	  = multiview_pass.colour.view;
		multiview_pass.descriptor.sampler	  = multiview_pass.sampler;
	}

	// depth/stencil FBO setup
	{
		VkImageCreateInfo image_ci = {
			.sType		 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext		 = nullptr,
			.flags		 = 0,
			.imageType	 = VK_IMAGE_TYPE_2D,
			.format		 = depthFormat,
			.extent		 = {width, height, 1},
			.mipLevels	 = 1,
			.arrayLayers = multiview_layers,
			.samples	 = VK_SAMPLE_COUNT_1_BIT,
			.tiling		 = VK_IMAGE_TILING_OPTIMAL,
			.usage		 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		};
		VK_CHECK_RESULT(vkCreateImage(device, &image_ci, nullptr, &multiview_pass.depth.image));

		VkMemoryRequirements memory_requirements;
		vkGetImageMemoryRequirements(device, multiview_pass.depth.image, &memory_requirements);

		VkMemoryAllocateInfo memory_ai = {
			.sType			 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize	 = memory_requirements.size,
			.memoryTypeIndex = vulkanDevice->getMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
		};

		VkImageSubresourceRange depth_stencil_subresource = {
			.aspectMask		= VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
			.baseMipLevel	= 0,
			.levelCount		= 1,
			.baseArrayLayer = 0,
			.layerCount		= multiview_layers,
		};

		VkImageViewCreateInfo depth_stencil_view = {
			.sType			  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext			  = nullptr,
			.flags			  = 0,
			.image			  = multiview_pass.depth.image,
			.viewType		  = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			.format			  = depthFormat,
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
			.flags			= 0,
			.format			= swapChain.colorFormat,
			.samples		= VK_SAMPLE_COUNT_1_BIT,
			.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp		= VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout	= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		// Depth attachment
		attachments[1] = {
			.flags			= 0,
			.format			= depthFormat,
			.samples		= VK_SAMPLE_COUNT_1_BIT,
			.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp		= VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_CLEAR,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout	= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		// Attachment references
		VkAttachmentReference colour_reference = {
			.attachment = 0,
			.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_reference = {
			.attachment = 1,
			.layout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};


		// Subpass dependencies
		VkSubpassDescription subpass_description = {
			.pipelineBindPoint		 = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount	 = 1,
			.pColorAttachments		 = &colour_reference,
			.pDepthStencilAttachment = &depth_reference,
		};

		VkSubpassDependency dependencies[2];
		dependencies[0] = {
			.srcSubpass		 = VK_SUBPASS_EXTERNAL,
			.dstSubpass		 = 0,
			.srcStageMask	 = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.dstStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask	 = VK_ACCESS_MEMORY_READ_BIT,
			.dstAccessMask	 = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
		};

		dependencies[1] = {
			.srcSubpass		 = 0,
			.dstSubpass		 = VK_SUBPASS_EXTERNAL,
			.srcStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask	 = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.srcAccessMask	 = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask	 = VK_ACCESS_MEMORY_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
		};

		// Bit mask for which view is rendering
		const uint32_t viewmask			= 0b00000011;
		const uint32_t correlation_mask = 0b00000011;

		// Multiview renderpass creation
		VkRenderPassMultiviewCreateInfo renderpass_multiview_ci = {
			.sType				  = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
			.pNext				  = nullptr,
			.subpassCount		  = 1,
			.pViewMasks			  = &viewmask,
			.correlationMaskCount = 1,
			.pCorrelationMasks	  = &correlation_mask,
		};


		VkRenderPassCreateInfo renderpass_ci = {
			.sType			 = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.pNext			 = &renderpass_multiview_ci,
			.flags			 = 0,
			.attachmentCount = 2,
			.pAttachments	 = attachments,
			.subpassCount	 = 1,
			.pSubpasses		 = &subpass_description,
			.dependencyCount = 2,
			.pDependencies	 = dependencies,
		};

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderpass_ci, nullptr, &multiview_pass.renderpass));
	}

	// Framebuffer creation
	{
		VkImageView fbo_attachments[] = {multiview_pass.colour.view, multiview_pass.depth.view};

		VkFramebufferCreateInfo fbo_ci = {
			.sType			 = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.pNext			 = nullptr,
			.flags			 = 0,
			.renderPass		 = multiview_pass.renderpass,
			.attachmentCount = 2,
			.pAttachments	 = fbo_attachments,
			.width			 = SERVERWIDTH,
			.height			 = SERVERHEIGHT,
			.layers			 = 1,
		};
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbo_ci, nullptr, &multiview_pass.framebuffer));
	}
}


void VulkanExample::buildCommandBuffers()
{
	VkCommandBufferBeginInfo cmdBufInfo =
		vks::initializers::commandBufferBeginInfo();

	VkClearValue clearValues[2];
	clearValues[0].color = defaultClearColor;
	clearValues[0].color = {{0.25f, 0.25f, 0.25f, 1.0f}};
	;
	clearValues[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo renderPassBeginInfo	 = vks::initializers::renderPassBeginInfo();
	renderPassBeginInfo.renderPass				 = renderPass;
	renderPassBeginInfo.renderArea.offset.x		 = 0;
	renderPassBeginInfo.renderArea.offset.y		 = 0;
	renderPassBeginInfo.renderArea.extent.width	 = width;
	renderPassBeginInfo.renderArea.extent.height = height;
	renderPassBeginInfo.clearValueCount			 = 2;
	renderPassBeginInfo.pClearValues			 = clearValues;

	const VkViewport viewport =
		vks::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
	const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

	for(int32_t i = 0; i < drawCmdBuffers.size(); ++i)
	{
		renderPassBeginInfo.framebuffer = frameBuffers[i];
		VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
		vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo,
							 VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
		vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
		// Bind scene matrices descriptor to set 0
		vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
								pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

		// POI: Draw the glTF scene
		glTFScene.draw(drawCmdBuffers[i], pipelineLayout);

		drawUI(drawCmdBuffers[i]);
		vkCmdEndRenderPass(drawCmdBuffers[i]);
		VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
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
	glTFScene.copyQueue	   = queue;

	size_t pos	   = filename.find_last_of('/');
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
	size_t indexBufferSize	= indexBuffer.size() * sizeof(uint32_t);
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

	// One ubo to pass dynamic data to the shader
	// Two combined image samplers per material as each material uses color and
	// normal maps
	std::vector<VkDescriptorPoolSize> poolSizes = {
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(glTFScene.materials.size()) * 2 + 1), // +1 for multiview
	};

	// One set for matrices and one per model image/texture
	const uint32_t maxSetCount					  = static_cast<uint32_t>(glTFScene.images.size()) + 1;
	VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
	VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

	// Descriptor set layout for passing matrices
	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
	};

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));

	// Descriptor set layout for passing material textures
	setLayoutBindings = {
		// Color map
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		// Normal map
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
	};

	descriptorSetLayoutCI.pBindings	   = setLayoutBindings.data();
	descriptorSetLayoutCI.bindingCount = 2;
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures));

	// Pipeline layout using both descriptor sets (set 0 = matrices, set 1 =
	// material)
	std::array<VkDescriptorSetLayout, 2> setLayouts = {descriptorSetLayouts.matrices, descriptorSetLayouts.textures};
	VkPipelineLayoutCreateInfo pipelineLayoutCI		= vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

	// We will use push constants to push the local matrices of a primitive to the
	// vertex shader
	VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
	// Push constant ranges are part of the pipeline layout
	pipelineLayoutCI.pushConstantRangeCount = 1;
	pipelineLayoutCI.pPushConstantRanges	= &pushConstantRange;
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

	// Descriptor set for scene matrices
	VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

	VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
	vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

	// Descriptor sets for materials
	for(uint32_t i = 0; i < glTFScene.materials.size(); i++)
	{
		const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.textures, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &glTFScene.materials[i].descriptorSet));

		VkDescriptorImageInfo colorMap	= glTFScene.getTextureDescriptor(glTFScene.materials[i].baseColorTextureIndex);
		VkDescriptorImageInfo normalMap = glTFScene.getTextureDescriptor(glTFScene.materials[i].normalTextureIndex);

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(glTFScene.materials[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &colorMap),
			vks::initializers::writeDescriptorSet(glTFScene.materials[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &normalMap),
			//vks::initializers::writeDescri
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}


	// Viewdisplay pipeline stuff
	{
		// descriptor set layout
		std::vector<VkDescriptorSetLayoutBinding> viewdisp_layout_bindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
		};

		VkDescriptorSetLayoutCreateInfo viewdisp_desc_layout = vks::initializers::descriptorSetLayoutCreateInfo(viewdisp_layout_bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &viewdisp_desc_layout, nullptr, &descriptorSetLayouts.viewdisp));

		// pipeline layout
		VkPipelineLayoutCreateInfo viewdisp_pl_layout_ci = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.viewdisp, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &viewdisp_pl_layout_ci, nullptr, &viewdisp_pipeline_layout));

		// Descriptors
		VkDescriptorSetAllocateInfo descriptorset_ai = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.viewdisp, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorset_ai, &viewdisp_descriptor_set))

		std::vector<VkWriteDescriptorSet> viewdisp_write_descriptor_sets = {
			vks::initializers::writeDescriptorSet(viewdisp_descriptor_set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &ubo)
		}
	}



	VkPipelineLayoutCreateInfo viewdisp_pipeline_layout_ci = vks::initializers::pipelineLayoutCreateInfo(&)
}

void VulkanExample::preparePipelines()
{
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
	VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
	VkPipelineColorBlendAttachmentState blendAttachmentStateCI	= vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
	VkPipelineColorBlendStateCreateInfo colorBlendStateCI		= vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
	VkPipelineDepthStencilStateCreateInfo depthStencilStateCI	= vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
	VkPipelineViewportStateCreateInfo viewportStateCI			= vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
	VkPipelineMultisampleStateCreateInfo multisampleStateCI		= vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
	const std::vector<VkDynamicState> dynamicStateEnables		= {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicStateCI				= vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);

	std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

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

	VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
	pipelineCI.pVertexInputState			= &vertexInputStateCI;
	pipelineCI.pInputAssemblyState			= &inputAssemblyStateCI;
	pipelineCI.pRasterizationState			= &rasterizationStateCI;
	pipelineCI.pColorBlendState				= &colorBlendStateCI;
	pipelineCI.pMultisampleState			= &multisampleStateCI;
	pipelineCI.pViewportState				= &viewportStateCI;
	pipelineCI.pDepthStencilState			= &depthStencilStateCI;
	pipelineCI.pDynamicState				= &dynamicStateCI;
	pipelineCI.stageCount					= static_cast<uint32_t>(shaderStages.size());
	pipelineCI.pStages						= shaderStages.data();

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

		materialSpecializationData.alphaMask	   = material.alphaMode == "MASK";
		materialSpecializationData.alphaMaskCutoff = material.alphaCutOff;

		// POI: Constant fragment shader material parameters will be set using
		// specialization constants
		std::vector<VkSpecializationMapEntry> specializationMapEntries = {
			vks::initializers::specializationMapEntry(0, offsetof(MaterialSpecializationData, alphaMask), sizeof(MaterialSpecializationData::alphaMask)),
			vks::initializers::specializationMapEntry(1, offsetof(MaterialSpecializationData, alphaMaskCutoff), sizeof(MaterialSpecializationData::alphaMaskCutoff)),
		};

		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(specializationMapEntries, sizeof(materialSpecializationData), &materialSpecializationData);
		shaderStages[1].pSpecializationInfo		= &specializationInfo;

		// For double sided materials, culling will be disabled
		rasterizationStateCI.cullMode = material.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &material.pipeline));
	}

	// Viewdisp pipelines setup
	{
		// First, query for multiview support stuff
		VkPhysicalDeviceFeatures2KHR multiview_device_features2{};
		VkPhysicalDeviceMultiviewFeaturesKHR multiview_extension_features{};
		multiview_extension_features.sType									= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR;
		multiview_device_features2.sType									= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
		multiview_device_features2.pNext									= &multiview_extension_features;
		PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
		vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &multiview_device_features2);
		std::cout << "Multiview features:" << std::endl;
		std::cout << "\tmultiview = " << multiview_extension_features.multiview << std::endl;
		std::cout << "\tmultiviewGeometryShader = " << multiview_extension_features.multiviewGeometryShader << std::endl;
		std::cout << "\tmultiviewTessellationShader = " << multiview_extension_features.multiviewTessellationShader << std::endl;
		std::cout << std::endl;

		VkPhysicalDeviceProperties2KHR device_properties2{};
		VkPhysicalDeviceMultiviewPropertiesKHR extension_properties{};
		extension_properties.sType												= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR;
		device_properties2.sType												= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
		device_properties2.pNext												= &extension_properties;
		PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR"));
		vkGetPhysicalDeviceProperties2KHR(physicalDevice, &device_properties2);
		std::cout << "Multiview properties:" << std::endl;
		std::cout << "\tmaxMultiviewViewCount = " << extension_properties.maxMultiviewViewCount << std::endl;
		std::cout << "\tmaxMultiviewInstanceIndex = " << extension_properties.maxMultiviewInstanceIndex << std::endl;

		// Viewdisplay for multiview
		VkPipelineShaderStageCreateInfo viewdisp_shader_stages[2];
		float multiview_array_layer								   = 0.0f;
		VkSpecializationMapEntry viewdisp_specialization_map_entry = {0, 0, sizeof(float)};
		VkSpecializationInfo viewdisp_specialization_info		   = {
			 .mapEntryCount = 1,
			 .pMapEntries	= &viewdisp_specialization_map_entry,
			 .dataSize		= sizeof(float),
			 .pData			= &multiview_array_layer,
		 };


		for(uint32_t i = 0; i < 2; i++)
		{
			viewdisp_shader_stages[0]					= loadShader(getShadersPath() + "gltfscenerendering/viewdisplay.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			viewdisp_shader_stages[1]					= loadShader(getShadersPath() + "gltfscenerendering/viewdisplay.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			viewdisp_shader_stages->pSpecializationInfo = &viewdisp_specialization_info;
			multiview_array_layer						= (float) i;

			VkPipelineVertexInputStateCreateInfo empty_input_state = vks::initializers::pipelineVertexInputStateCreateInfo();
			pipelineCI.pVertexInputState						   = &empty_input_state;
			pipelineCI.layout									   = viewdisp_pipeline_layout;
			pipelineCI.renderPass								   = renderPass;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &viewdisp_pipelines[i]));
		}
	}
}

void VulkanExample::prepareUniformBuffers()
{
	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&shaderData.buffer, sizeof(shaderData.values)));
	VK_CHECK_RESULT(shaderData.buffer.map());
	updateUniformBuffers();
}

void VulkanExample::updateUniformBuffers()
{
	shaderData.values.projection = camera.matrices.perspective;
	shaderData.values.view		 = camera.matrices.view;
	shaderData.values.viewPos	 = camera.viewPos;
	memcpy(shaderData.buffer.mapped, &shaderData.values,
		   sizeof(shaderData.values));
}

void VulkanExample::prepare()
{
	VulkanExampleBase::prepare();
	loadAssets();
	setup_multiview();
	prepareUniformBuffers();
	setupDescriptors();
	preparePipelines();
	buildCommandBuffers();
	prepared = true;
}

void VulkanExample::render()
{
	renderFrame();
	if(camera.updated)
	{
		updateUniformBuffers();
	}
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