/*
Name: Open_file_and_write.c
Author: DnH500. Darknighthunter500@gmail.com
Date: 02/01/05
Description: Opens file by append example.
*/
#include <stdio.h>
#include <time.h>

int main()
{
    clock_t start = clock();

    FILE *file;
    int i;

    file = fopen("times.txt","a+"); /* apend file (add text to
    a file or create a file if it does not exist.*/
    for ( i = 0; i < 10; ++i )
    {
        fprintf(file,"%d time: %f\n", i, ((double)clock()-start)/CLOCKS_PER_SEC); /*writes*/
    }
    fclose(file); /*done!*/


    return 0;
}
