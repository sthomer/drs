include make.inc .master.inc

SRCS := $(wildcard ./src/*/*.f90)
OBJS := $(SRCS:.f90=.o)

all: lib$(LIBNAME).$(SLIB)

install: lib$(LIBNAME).$(SLIB)
	@mkdir -p $(INSTALLDIR)/$(LIBNAME)/lib
	@cp ./lib$(LIBNAME).$(SLIB) $(INSTALLDIR)/$(LIBNAME)/lib

example%:
	@$(MAKE) $@ -C ./example

test%:
	@$(MAKE) $@ -C ./test

benchmark%:
	@$(MAKE) $@ -C ./benchmark

lib$(LIBNAME).$(SLIB): srcs
	$(FC) $(FFLAGS) -shared -o lib$(LIBNAME).$(SLIB) $(OBJS)

srcs:
	@$(MAKE) $@ -C ./src

uninstall: clean
	@rm -rf $(INSTALLDIR)/$(LIBNAME)

clean:
	@$(MAKE) clean -C ./src
	@$(MAKE) clean -C ./example
	@$(MAKE) clean -C ./test
	@$(MAKE) clean -C ./benchmark
	@rm -f lib$(LIBNAME).$(SLIB)


