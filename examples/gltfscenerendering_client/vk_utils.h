#ifndef VK_UTILS_H
#define VK_UTILS_H

#include <sys/time.h>

#include "vulkanexamplebase.h"


namespace vku
{
	VkCommandBuffer begin_command_buffer(VkDevice logical_device, VkCommandPool command_pool)
	{
		VkCommandBufferAllocateInfo cmdbuf_ai = {
			.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool		= command_pool,
			.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};

		VkCommandBufferBeginInfo cmdbuf_bi = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		VkCommandBuffer command_buffer;
		vkAllocateCommandBuffers(logical_device, &cmdbuf_ai, &command_buffer);
		vkBeginCommandBuffer(command_buffer, &cmdbuf_bi);

		return command_buffer;
	}

	void end_command_buffer(VkDevice logical_device, VkQueue queue, VkCommandPool command_pool, VkCommandBuffer command_buffer)
	{
		vkEndCommandBuffer(command_buffer);

		VkSubmitInfo submit_info = {
			.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers	= &command_buffer,
		};

		vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);
		vkFreeCommandBuffers(logical_device, command_pool, 1, &command_buffer);
	}

	inline VkImageSubresourceRange imageSubresourceRange(VkImageAspectFlags aspectMask, uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount)
	{
		VkImageSubresourceRange image_subresource_range = {
			.aspectMask		= aspectMask,
			.baseMipLevel	= baseMipLevel,
			.levelCount		= levelCount,
			.baseArrayLayer = baseArrayLayer,
			.layerCount		= layerCount,
		};

		return image_subresource_range;
	}


	VkImageMemoryBarrier imageMemoryBarrier(VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex, VkImage image, VkImageSubresourceRange subresourceRange)
	{
		VkImageMemoryBarrier image_memory_barrier = {
			.sType				 = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask		 = srcAccessMask,
			.dstAccessMask		 = dstAccessMask,
			.oldLayout			 = oldLayout,
			.newLayout			 = newLayout,
			.srcQueueFamilyIndex = srcQueueFamilyIndex,
			.dstQueueFamilyIndex = dstQueueFamilyIndex,
			.image				 = image,
			.subresourceRange	 = subresourceRange,
		};

		return image_memory_barrier;
	}


	void transition_image_layout(VkDevice logical_device, VkQueue queue, VkCommandPool command_pool, VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout)
	{
		VkCommandBuffer command_buffer			  = vku::begin_command_buffer(logical_device, command_pool);
		VkImageSubresourceRange subresource_range = vku::imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
		VkImageMemoryBarrier barrier			  = vku::imageMemoryBarrier(0, 0, old_layout, new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, subresource_range);

		VkPipelineStageFlags src_stage;
		VkPipelineStageFlags dst_stage;

		if(old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			src_stage			  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dst_stage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}

		else if(old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			src_stage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dst_stage			  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}

		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		// the pipeline stage to submit, pipeline stage to wait on
		vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
		vku::end_command_buffer(logical_device, queue, command_pool, command_buffer);
	}

	void transition_image_layout(VkDevice logical_device, VkCommandPool command_pool, VkCommandBuffer command_buffer, VkImage image, VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask, VkImageLayout old_layout, VkImageLayout new_layout, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask)
	{
		VkImageSubresourceRange subresource_range = vku::imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
		VkImageMemoryBarrier barrier			  = vku::imageMemoryBarrier(src_access_mask, dst_access_mask, old_layout, new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, subresource_range);

		// the pipeline stage to submit, pipeline stage to wait on
		vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	void rgba_to_rgb(const uint8_t *__restrict__ in, uint8_t *__restrict__ out, size_t len)
	{
		for(size_t i = 0, j = 0; i < len; i += 4, j += 3)
		{
			out[j + 0] = in[i + 0];
			out[j + 1] = in[i + 1];
			out[j + 2] = in[i + 2];
		}
	}


	void rgb_to_rgba(const uint8_t *__restrict__ in, uint8_t *__restrict__ out)
	{
		size_t len = 320 * 240 * 4;


		for(size_t i = 0, j = 0; i < len; i += 4, j += 3)
		{
			out[i + 0] = in[j + 2];
			out[i + 1] = in[j + 1];
			out[i + 2] = in[j + 0];
			out[i + 3] = 255;
		}
	}

	float time_difference(timeval start, timeval end)
	{
		return ((end.tv_sec - start.tv_sec) * 1000.0f) + ((end.tv_usec - start.tv_usec) / 1000.0f);
	}
} // namespace vku

#endif