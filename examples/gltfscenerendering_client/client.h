#ifndef CLIENT_H
#define CLIENT_H


#include <arpa/inet.h>
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
			//.sin_addr	= inet_addr("192.168.1.6"),
		};
		//inet_aton("192.168.1.6", (in_addr*) &server_address.sin_addr.s_addr);

		inet_pton(AF_INET, "192.168.1.6", &(server_address.sin_addr));
		std::cout << "sin_addr: " << server_address.sin_addr.s_addr << "\n";
		printf("Connected to server\n");

		int connect_result = connect(socket_fd, (sockaddr *) &server_address, sizeof(server_address));
		if(connect_result == -1)
		{
			std::string filename = "/sdcard/gltfscenerendering_client_log.txt";
			std::ofstream file(filename, std::ios::out | std::ios::binary);
			file << "Could not connect to server\n"
				<< "PORT: " << server_address.sin_port << "\t" << "IP: " << server_address.sin_addr.s_addr << "\n";
			file.close();
			throw std::runtime_error("Could not connect to server");
		}
	}
};

#endif