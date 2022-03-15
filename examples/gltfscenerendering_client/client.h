#ifndef CLIENT_H
#define CLIENT_H

#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>


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

		// Set socket non blocking
		/*int nonblockinresult = fcntl(socket_fd, F_GETFL) & O_NONBLOCK;

		if(nonblockinresult == -1)
		{
			throw std::runtime_error("Could not set client socket file descriptor to non blocking");
		}*/
	}

	void connect_to_server(int port)
	{
		sockaddr_in server_address = {
			.sin_family = AF_INET,
			.sin_port	= htons(static_cast<in_port_t>(port)),
			//.sin_addr	= inet_addr("192.168.0.2"),
		};

		//inet_pton(AF_INET, "silo.remexre.xyz", &(server_address.sin_addr));

		int connect_result = connect(socket_fd, (sockaddr *) &server_address, sizeof(server_address));
		if(connect_result == -1)
		{
			throw std::runtime_error("Could not connect to server");
		}
	}
};

#endif