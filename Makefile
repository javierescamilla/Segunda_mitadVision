#LCCLIBS+= -larsal -lardiscovery -larcontroller -lncurses 
LXXLIBS+= -larsal -lardiscovery -larcontroller -lavcodec  -lswscale  -lavutil -lavformat -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_imgproc -lopencv_videoio -lpthread

OBJS=$(patsubst %.cpp,%.o,$(shell find src/ | grep .cpp))
#OBJS+=$(patsubst %.c,%.o,$(wildcard src/*.c) $(wildcard src/*/*.c))
TARGETDIR = ./bin
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

all: dirs $(OBJS)
	@echo Linking everything together...
	@g++ $(OBJS) -o $(TARGETDIR)/main $(LCCLIBS) $(LXXLIBS)
	@echo Done

dirs:
	@mkdir -p $(TARGETDIR)


#.c.o:	
#	gcc -c $<  -o $@  $(LCCLIBS)

.cpp.o:
	@echo Compiling $<...
	@g++ -c -std=c++11 $<  -o $@  $(LXXLIBS) 

install:
	wget -O Installx64.zip "https://drive.google.com/uc?export=download&id=1ZiTAGkTFdDkEJ7CYhXHNIhfJgFtE_uw_" 
	unzip Installx64.zip
	cp -r  Install/usr /
	rm -r Install
	rm Installx64.zip
	ldconfig

clean:
	@$(RM) -rf $(OBJS) bin
