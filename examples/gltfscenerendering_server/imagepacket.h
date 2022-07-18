#ifndef IMAGE_PACKET_H
#define IMAGE_PACKET_H

#include "VulkanBuffer.h"
#include "vulkanexamplebase.h"

struct ImagePacket
{
	vks::Buffer buffer;
	void *data;
	size_t num_bytes;
};



#endif