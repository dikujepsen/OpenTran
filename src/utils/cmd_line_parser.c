#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void check_argument(char* arg) {

    if (arg[0] != '-') {
        printf("Error in argument %s. Arguments start with '-'\n", arg);
    }
}

char *strdup (const char *s) {
    char *d = malloc (strlen (s) + 1);
    if (d == NULL) return NULL;
    strcpy (d,s);
    return d;
}

struct parameter {
    char * name;
    char * val1;
    char * val2;
};



void split_token(char* arg, char* split, char** out_tokens) {
    if (arg != NULL) {
        char *arg_copy = strdup(arg);
        out_tokens[0] = NULL;
        out_tokens[1] = NULL;
        char *token;

        token = strtok(arg_copy, split);
        printf( "my %s\n", token );
        int t_i = 0;
        while( token != NULL && t_i < 2)
        {
          printf( " %s\n", token );
          out_tokens[t_i] = strdup(token);
          t_i = t_i + 1;
          printf( "my2 %s\n", token );
          token = strtok(NULL, split);
          printf( "my3 %s\n", token );
        }
    }
}


struct parameter parse_arg(char* arg) {
    char *tokens[2];
    split_token(arg, "=", tokens);

    struct parameter retval;
    retval.name = tokens[0];
    char *tokens2[2];
    split_token(tokens[1], ",", tokens2);
    retval.val1 = tokens2[0];
    retval.val2 = tokens2[1];

    return retval;
}


int main(int argc, char** argv)
{
    printf("\n");

	for (int i = 1; i < argc; i++)
	{

        check_argument(argv[i]);
		printf("%s\n", argv[i]);
	}

	const char *my_str_literal = argv[1];
    char *str_copy = strdup(my_str_literal);

    char *tokens[2];
    char *token;
    const char s[2] = "=";
    token = strtok(str_copy, s);
    int t_i = 0;
    while( token != NULL )
    {
      printf( " %s\n", token );
      tokens[t_i] = strdup(token);
      t_i = t_i + 1;
      token = strtok(NULL, s);
    }

    printf("%d\n", t_i);
    for(int i = 0; i < t_i; i++) {
        printf("HER1\n");
        printf("Token %s\n", tokens[i]);

    }
    printf("HER2\n");

    free(str_copy);
    str_copy = strdup(my_str_literal);
    struct parameter arg = parse_arg(str_copy);
    printf("Token1 %s\n", arg.name);
    printf("Token2 %s\n", arg.val1);
    printf("Token3 %s\n", arg.val2);

    free(str_copy);


	printf("\n");
	return 0;
}