CC = g++
PROJECT = vo
SRC = src/main.cpp src/pose_estimation_node.cpp src/vo.cpp src/feature.cpp src/matrix.cpp src/bucket.cpp src/utils.cpp include/vo.h include/bucket.h include/feature.h include/matrix.h include/pose_estimation_node.h include/utils.h
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d `pkg-config --cflags --libs opencv4`
CFLAGS = -Iinclude -Wall
$(PROJECT) : $(SRC)
	$(CC) $(SRC) -g $(CFLAGS) -o $(PROJECT) $(LIBS)
