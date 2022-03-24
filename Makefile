CC = g++
PROJECT = vo
SRC = src/main.cpp src/vo.cpp src/feature_set.cpp include/vo.h
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d `pkg-config --cflags --libs opencv4`
CFLAGS = -Iinclude -Wall
$(PROJECT) : $(SRC)
	$(CC) $(SRC) -g $(CFLAGS) -o $(PROJECT) $(LIBS)
