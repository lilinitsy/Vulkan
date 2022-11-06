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

#include <sys/time.h>

#include "gltfscenerendering_client.h"

/*
		Vulkan glTF scene class
*/

pthread_mutex_t gpu_map_lock;

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
	VulkanExampleBase(false, CLIENTWIDTH, CLIENTHEIGHT)
{
	title		 = "glTF scene rendering";
	camera.type	 = Camera::CameraType::firstperson;
	camera.flipY = true;
	camera.setPosition(glm::vec3(2.2f, -2.0f, 0.25f));
	camera.setRotation(glm::vec3(-180.0f, -90.0f, 0.0f));
	camera.movementSpeed = 4.0f;
}

VulkanExample::~VulkanExample()
{
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptor_set_layouts.matrices, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptor_set_layouts.textures, nullptr);
	shaderData.buffer.destroy();
}

void VulkanExample::getEnabledFeatures()
{
	enabledFeatures.samplerAnisotropy = deviceFeatures.samplerAnisotropy;
}


void VulkanExample::buildCommandBuffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

	VkClearValue clearValues[2];
	clearValues[0].color		= defaultClearColor;
	clearValues[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo renderPassBeginInfo	 = vks::initializers::renderPassBeginInfo();
	renderPassBeginInfo.renderPass				 = renderPass;
	renderPassBeginInfo.renderArea.offset.x		 = 0;
	renderPassBeginInfo.renderArea.offset.y		 = 0;
	renderPassBeginInfo.renderArea.extent.width	 = width;
	renderPassBeginInfo.renderArea.extent.height = height;
	renderPassBeginInfo.clearValueCount			 = 2;
	renderPassBeginInfo.pClearValues			 = clearValues;

	VkViewport viewport = vks::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
	VkRect2D scissor	= vks::initializers::rect2D(width, height, 0, 0);

	for(int32_t i = 0; i < drawCmdBuffers.size(); ++i)
	{
		renderPassBeginInfo.framebuffer = frameBuffers[i];
		VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

		vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
		vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

		// Bind scene matrices descriptor to set 0
		vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

		// POI: Draw the glTF scene
		glTFScene.draw(drawCmdBuffers[i], pipeline_layout);

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

	// ========================================================================
	//							SETUP FOR POOL
	// ========================================================================
	// One ubo to pass dynamic data to the shader
	// Two combined image samplers per material as each material uses color and
	// normal maps
	std::vector<VkDescriptorPoolSize> poolSizes = {
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(glTFScene.materials.size()) * 2), // +1 for multiview
	};

	// One set for matrices and one per model image/texture
	const uint32_t maxSetCount					  = static_cast<uint32_t>(glTFScene.images.size()) + 1;
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

	descriptorSetLayoutCI.pBindings	   = setLayoutBindings.data();
	descriptorSetLayoutCI.bindingCount = 2;
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptor_set_layouts.textures));


	// ========================================================================
	//							GLTF PIPELINE LAYOUT
	// ========================================================================


	// Pipeline layout using both descriptor sets (set 0 = matrices, set 1 =
	// material)
	std::array<VkDescriptorSetLayout, 2> setLayouts = {descriptor_set_layouts.matrices, descriptor_set_layouts.textures};
	VkPipelineLayoutCreateInfo pipelineLayoutCI		= vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

	// We will use push constants to push the local matrices of a primitive to the
	// vertex shader
	VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
	// Push constant ranges are part of the pipeline layout
	pipelineLayoutCI.pushConstantRangeCount = 1;
	pipelineLayoutCI.pPushConstantRanges	= &pushConstantRange;
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipeline_layout));

	// Descriptor set for scene matrices
	VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptor_set_layouts.matrices, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set));

	VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptor_set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
	vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

	// Descriptor sets for materials
	for(uint32_t i = 0; i < glTFScene.materials.size(); i++)
	{
		const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptor_set_layouts.textures, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &glTFScene.materials[i].descriptorSet));

		VkDescriptorImageInfo colorMap	= glTFScene.getTextureDescriptor(glTFScene.materials[i].baseColorTextureIndex);
		VkDescriptorImageInfo normalMap = glTFScene.getTextureDescriptor(glTFScene.materials[i].normalTextureIndex);

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(glTFScene.materials[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &colorMap),
			vks::initializers::writeDescriptorSet(glTFScene.materials[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &normalMap),
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}
}

void VulkanExample::preparePipelines()
{
	// ========================================================================
	//							GENERAL GRAPHICS PIPELINE SETUP
	// ========================================================================

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

	VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipeline_layout, renderPass, 0);
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


	shaderStages[0] = loadShader(getShadersPath() + "gltfscenerendering_client/multiview.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
	shaderStages[1] = loadShader(getShadersPath() + "gltfscenerendering_client/multiview.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);


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
		rasterizationStateCI.cullMode = material.doubleSided ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &material.pipeline));
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
	float wd2		  = zNear * tan(glm::radians(fov / 2.0f));
	float ndfl		  = zNear / focalLength;
	float left, right;
	float top	 = wd2;
	float bottom = -wd2;

	glm::vec3 camFront;
	camFront.x		   = -cos(glm::radians(camera.rotation.x)) * sin(glm::radians(camera.rotation.y));
	camFront.y		   = -sin(glm::radians(camera.rotation.x));
	camFront.z		   = cos(glm::radians(camera.rotation.x)) * cos(glm::radians(camera.rotation.y));
	camFront		   = glm::normalize(camFront);
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
	shaderData.values.view[0]		= rotM * transM;

	// Right eye
	left  = -aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;
	right = aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;

	transM = glm::translate(glm::mat4(1.0f), camera.position + camRight * (eyeSeparation / 2.0f));

	shaderData.values.projection[1] = glm::frustum(left, right, bottom, top, zNear, zFar);
	shaderData.values.view[1]		= rotM * transM;

	memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));
}


void transition_image_layout(VkCommandBuffer command_buffer, VkImage image, VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask, VkImageLayout old_layout, VkImageLayout new_layout, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask)
{
	VkImageSubresourceRange subresource_range = vku::imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
	VkImageMemoryBarrier barrier			  = vku::imageMemoryBarrier(src_access_mask, dst_access_mask, old_layout, new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, subresource_range);

	// the pipeline stage to submit, pipeline stage to wait on
	vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}


void VulkanExample::prepare()
{
	if(pthread_mutex_init(&gpu_map_lock, nullptr) != 0)
	{
		printf("mutex initialization unsuccessful\n");
		return;
	}
	VulkanExampleBase::prepare();
	loadAssets();
	prepareUniformBuffers();
	setupDescriptors();
	preparePipelines();
	VulkanExample::buildCommandBuffers();
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
			VulkanExample::buildCommandBuffers();
		}
		ImGui::SameLine();
		if(overlay->button("None"))
		{
			std::for_each(glTFScene.nodes.begin(), glTFScene.nodes.end(),
						  [](VulkanglTFScene::Node &node)
						  { node.visible = false; });
			VulkanExample::buildCommandBuffers();
		}
		ImGui::NewLine();

		// POI: Create a list of glTF nodes for visibility toggle
		ImGui::BeginChild("#nodelist", ImVec2(200.0f, 340.0f), false);
		for(auto &node : glTFScene.nodes)
		{
			if(overlay->checkBox(node.name.c_str(), &node.visible))
			{
				VulkanExample::buildCommandBuffers();
			}
		}
		ImGui::EndChild();
	}
}

VULKAN_EXAMPLE_MAIN()