#ifndef IMAGE_PACKET_H
#define IMAGE_PACKET_H

#include "vulkanexamplebase.h"

struct ImagePacket
{
	VkImage image;
	VkDeviceMemory memory;
	VkSubresourceLayout subresource_layout;
	char *data;

	void map_memory(VkDevice logical_device)
	{
		vkMapMemory(logical_device, memory, 0, VK_WHOLE_SIZE, 0, (void **) &data);
		data += subresource_layout.offset;
		vkUnmapMemory(logical_device, memory);
	}

	void destroy(VkDevice logical_device)
	{
		vkDestroyImage(logical_device, image, nullptr);
		vkFreeMemory(logical_device, memory, nullptr);
	}
};



#endif