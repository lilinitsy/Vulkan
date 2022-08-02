#ifndef SERVER_H
#define SERVER_H

#include <netinet/in.h>
#include <sys/socket.h>

#include "vk_utils.h"


struct Server
{
	int socket_fd[2];
	int client_fd[2];

	Server()
	{
		for(uint32_t i = 0; i < 2; i++)
		{
			// Create socket
			socket_fd[i] = socket(AF_INET, SOCK_STREAM, 0);
			if(socket_fd[i] == 0)
			{
				throw std::runtime_error("Socket creation failed");
			}
		}
	}

	void connect_to_client(const uint32_t PORT[])
	{
		for(uint32_t i = 0; i < 2; i++)
		{
			// define the address struct to be for TCP using this port
			sockaddr_in address = {
				.sin_family = AF_INET,
				.sin_port   = static_cast<in_port_t>(htons((int) PORT[i])),
			};

			// bind to socket
			int binding = bind(socket_fd[i], (sockaddr *) &address, sizeof(address));
			printf("Binding on port %d\n", PORT[i]);
			if(binding == -1)
			{
				std::string port_socketnum_str = std::to_string(i) + "	 " + std::to_string(PORT[i]);
				throw std::runtime_error("Bind to socket failed: " + port_socketnum_str);
			}

			// Listen for a client to connect
			listen(socket_fd[i], 1);
			printf("Listening for client on port %d\n", PORT[i]);
		}

		for(uint32_t i = 0; i < 2; i++)
		{
			// Accept a connection from a client
			client_fd[i] = accept(socket_fd[i], nullptr, nullptr);

			printf("Client connection accepted on port %d\n\n", PORT[i]);
		}
	}
};


#endif