# The following two variable will be commandline determined by TA
# For testing, you could uncomment them.
SRIPATH ?= /data/DSP_HW3/103_2/srilm-1.5.10
MACHINE_TYPE ?= i686-m64
LM ?= bigram.lm

SRIPATH = /mnt/d/ntu/DSP/dsp_hw3/srilm-1.5.10
MACHINE_TYPE = i686-m64
LM = bigram.lm

CXX = g++
CXXFLAGS = -O3 -std=c++11 -I$(SRIPATH)/include -w
vpath lib%.a $(SRIPATH)/lib/$(MACHINE_TYPE)

TARGET = mydisambig
SRC = mydisambig.cpp
OBJ = $(SRC:.cpp=.o)
TO = ZhuYin-Big5.map
FROM = Big5-ZhuYin.map
.PHONY: all clean map run

all: $(TARGET)

$(TARGET): $(OBJ) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<
run:
	mkdir result2 -p
	for i in $(shell seq 1 10) ; do \
            echo "Running $$i.txt"; \
            ./$(TARGET) -text testdata/$$i.txt -map $(TO) -lm $(LM) -order 2 > result2/$$i.txt; \
	done;
map: mapping.cpp
	echo "Mapping!"
	$(CXX) $< -o mapping
	./mapping $(FROM) $(TO)
clean:
	$(RM) $(OBJ) $(TARGET) $(TO)

