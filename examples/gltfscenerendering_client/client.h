#ifndef CLIENT_H
#define CLIENT_H


#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>


struct Client
{
	int socket_fd;

	Client()
	{
		socket_fd = socket(AF_INET, SOCK_STREAM, 0);
		if(socket_fd == -1)
		{
			throw std::runtime_error("Could not create a socket");
		}
		uint32_t optval = 1;
		setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
	}

	void connect_to_server(const uint32_t PORT)
	{

		sockaddr_in server_address = {
			.sin_family = AF_INET,
			.sin_port	= htons(static_cast<in_port_t>(PORT)),
			//.sin_addr	= inet_addr("192.168.1.6"),
		};
		//inet_aton("192.168.1.6", (in_addr*) &server_address.sin_addr.s_addr);

		inet_pton(AF_INET, "192.168.1.6", &(server_address.sin_addr));
		printf("Connected to server\n");


		int connect_result = connect(socket_fd, (sockaddr *) &server_address, sizeof(server_address));
		if(connect_result == -1)
		{

			printf("COULD NOT CONNECT TO SERVER on port %d\n", PORT);
			throw std::runtime_error("Could not connect to server");
		}
		printf("Connected to server on port %d\n", PORT);
	}

};

#endif