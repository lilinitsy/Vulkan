#ifndef CLIENT_H
#define CLIENT_H


#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>


struct Client
{
	int socket_fd[2];

	Client()
	{
		for(uint32_t i = 0; i < 2; i++)
		{
			socket_fd[i] = socket(AF_INET, SOCK_STREAM, 0);
			if(socket_fd[i] == -1)
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
	}

	void connect_to_server(const uint32_t PORT[])
	{
		for(uint32_t i = 0; i < 2; i++)
		{
			sockaddr_in server_address = {
				.sin_family = AF_INET,
				.sin_port	= htons(static_cast<in_port_t>(PORT[i])),
				//.sin_addr	= inet_addr("192.168.1.6"),
			};
			//inet_aton("192.168.1.6", (in_addr*) &server_address.sin_addr.s_addr);

			inet_pton(AF_INET, "192.168.1.6", &(server_address.sin_addr));
			printf("Connected to server\n");
			

			int connect_result = connect(socket_fd[i], (sockaddr *) &server_address, sizeof(server_address));
			if(connect_result == -1)
			{

				printf("COULD NOT CONNECT TO SERVER on port %d\n", PORT[i]);
				throw std::runtime_error("Could not connect to server");
			}
			printf("Connected to server on port %d\n", PORT[i]);
		}
	}
};

#endif