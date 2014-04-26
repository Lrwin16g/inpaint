# Makefile

CC = g++
CFLAGS = -O
INCPATH += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

EXEDIR = ./bin
OBJDIR = ./obj
SRCDIR = ./src

TARGET1 = $(EXEDIR)/inpaint
TARGET2 = $(EXEDIR)/train
TARGET3 = $(EXEDIR)/psnr

OBJ1 = $(OBJDIR)/inpaint.o $(OBJDIR)/gFoE.o
OBJ2 = $(OBJDIR)/train.o $(OBJDIR)/gFoE.o
OBJ3 = $(OBJDIR)/psnr.o

all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): $(OBJ1)
	$(CC) $(LIBS) -o $(TARGET1) $^

$(TARGET2): $(OBJ2)
	$(CC) $(LIBS) -o $(TARGET2) $^

$(TARGET3): $(OBJ3)
	$(CC) $(LIBS) -o $(TARGET3) $^

$(OBJDIR)/inpaint.o: $(SRCDIR)/inpaint.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/train.o: $(SRCDIR)/train.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/gFoE.o: $(SRCDIR)/gFoE.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/psnr.o: $(SRCDIR)/psnr.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(OBJDIR)/*.o
