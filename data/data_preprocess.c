/*
    Caoze Huang, chuan134@jh.edu
    Algorithm For File Preprocessing For Raw Raman Spectra

    For now, it simply converts text files to csv format
    To use, cd over, do make, and enter $ ./preprocess <input.txt>
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Helpers Initialization
char* Out_Name(char inputName) {}

// Main
int main(int argc, char *argv[]) {
    // Check if a existing file argument is passed
    if (argc < 2) {
        printf("Error: No file was provided.\n");
        printf("Usage: %s <your_file.txt>\n", argv[0]);
        return 1; 
    }

    // File I/O
    char inputName[] = argv[1];

    // Open input file
    FILE *inputFile = fopen(inputName, "r");
    if (inputFile == NULL) {
        perror("Error opening input.txt");
        return 1;
    }

    char outputName[] = Out_Name(inputName);

    // Open the output CSV file for writing
    FILE *outputFile = fopen(outputName, "w");
    if (outputFile == NULL) {
        perror("Error opening output.csv");
        fclose(inputFile);
        return 1;
    }
    // Print header
    char header[] = "spectra, amplitude\n";
        fputs(header, outputFile);
    
    // Read and print
    char ch;    
    while ((ch = fgetc(inputFile)) != EOF) {
        // Replace spaces or tabs with commas
        if (ch == ' ' || ch == '\t') {
            fputc(',', outputFile);
        } else {
            fputc(ch, outputFile);
        }
    }

    // Close files to free up resources
    fclose(inputFile);
    fclose(outputFile);

    printf("Successfully converted input.txt to output.csv!\n");

    // Freeing and cleaning
    free(outputName); 
    return 0;
}

char* Out_Name(char* inputName) {
    // Dynamically allocate memory for output name (+1 for null terminator)
    char* outputName = malloc(strlen(inputName) + 5); 
    if (outputName == NULL) {
        return NULL; // Handle allocation failure
    }

    // Cycle through and extract char
    int j = 0;
    for (int i = 0; i < strlen(inputName); i++) {
        char c = inputName[i];
        if (c != ' ') {
            outputName[j++] = c;
        }
    }
    
    // Add .csv extension and null terminator
    outputName[j] = '\0';
    strcat(outputName, ".csv"); 

    return outputName;
}