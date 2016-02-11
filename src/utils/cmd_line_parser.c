#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>


void check_argument(char* arg) {

    if (arg[0] != '-') {
        printf("Error in argument %s. Arguments start with '-'\n", arg);
    }
}

int main(int argc, char** argv)
{
    printf("\n");
	int i = 0;
	int first_idx = 0;
	for (i = 0; i < argc; i++)
	{
	    if (argv[i] == " ") {
	    first_idx = i;
	    break;
	    }
	}

	for (i = 1; i < argc; i++)
	{

        check_argument(argv[i]);
		printf("%s\n", argv[i]);
	}

	const char *my_str_literal = argv[1];
    char *str_copy = strdup(my_str_literal);
//    char *d = malloc (strlen (my_str_literal) + 1);
//    if (d == NULL) return NULL;
//    strcpy (d, my_str_literal);
//    char *str_copy = d;

    char *token;
    while ((token = strsep(&str_copy, "="))) {
        printf("Token %s\n", token);
    }

    free(str_copy);


	printf("\n");
	return 0;
}