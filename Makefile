install:
	make -C lib/eiscor install
	FC="gfortran" python -m numpy.f2py -c -L$(PWD)/lib/eiscor -leiscor \
	   lib/eiscor/src/complex_double/z_poly_roots.f90 -m eiscor \
	   --backend meson
	cp lib/eiscor/*dylib* .

clean:
	make -C lib/eiscor clean
	rm -f eiscor*.so libeiscor*
