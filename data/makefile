# Preprocessing Makefile
CC=gcc
CFLAGS=-std=c99 -pedantic -Wall -Wextra
TARGET = preprocess

all: $(TARGET)

# Links together files needed to create executable
$(TARGET): data_preprocess.c
	$(CC) $(CFLAGS) -o $(TARGET) data_preprocess.c

# Rule to clean up the compiled binary
clean:
	rm -f $(TARGET)