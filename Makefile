all:RNN
#opt = -O3 -g -pg -shared -Wl,-soname,pyrnn.so -lboost_python -lpython2.7 -fPIC
#opt = -O3 -g -pg -shared -lpython2.7 -fPIC
opt = -O3 -shared -lpython2.7 -fPIC
opt_end = -lpython2.7 -fPIC
CXXFLAGS = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include
LDLIBPATH = 

RNN: rnn.o
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -o RNN.so rnn.o $(opt_end)

rnn.o: main.cxx
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -c main.cxx -o rnn.o $(opt_end)

rnn: rnn.o
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -o rnn main.cxx $(opt_end)


clean:
	rm -rf *.o *.so
