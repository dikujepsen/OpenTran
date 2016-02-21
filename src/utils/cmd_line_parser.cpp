#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
    printf("\n");

	for (int i = 1; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}

	const char *my_str_literal = argv[1];


	printf("\n");
	return 0;
}