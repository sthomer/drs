install:
	make -C eiscor install
	FC="gfortran" python -m numpy.f2py -c -L$(PWD)/eiscor -leiscor \
	   eiscor/src/complex_double/z_poly_roots.f90 -m eiscor \
	   --backend meson
	cp eiscor/*dylib* .

clean:
	make -C eiscor clean
	rm -rf eiscor*.so libeiscor* __pycache__
