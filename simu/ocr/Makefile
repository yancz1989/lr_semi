CXXFLAGS = -O2 -Llibjt/lib/gcc4.4.6_Linux64 -Llibjt/lib/gcc4.4.6_Linux64/thirdparty -Ilibjt/inc/ -liRead_Table -liRead_Binarize -liRead_Deskew -liRead_Image -liRead_Layout -liRead_Recog -liRead_Util -liReadAPI -liReadExtAPI -liReadText -ljtCodeLib -ljtImageLib -lmkl_rt -lopencv_core -lopencv_imgproc

ALL:
	g++ -o ocr ocr.cpp $(CXXFLAGS)