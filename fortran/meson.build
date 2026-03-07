py = import('python').find_installation(pure: false)

numpy_inc = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

f2py_inc = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

project_inc = include_directories('eiscor/include')

fortran_src = []
subdir('eiscor/src')

fortran_wrap_src = files(
  'eiscor/src/complex_double/z_poly_roots.f90',
)

f2py_gen = custom_target(
  'eiscor_f2py_wrappers',
  input: fortran_wrap_src,
  output: [
    '_eiscor_extmodule.c',
    '_eiscor_ext-f2pywrappers.f',
  ],
  command: [
    py, '-m', 'numpy.f2py',
    '@INPUT@',
    'only:', 'z_poly_roots', ':',
    '-m', '_eiscor_ext',
    '--lower',
    '--build-dir', '@OUTDIR@'
  ]
)

py.extension_module(
  '_eiscor_ext',
  [
    f2py_gen,
    fortran_src,
    f2py_inc / 'fortranobject.c',
  ],
  include_directories: [project_inc],
  c_args: [
    '-I' + numpy_inc,
    '-I' + f2py_inc,
    '-Wno-missing-field-initializers',
    '-Wno-unused-parameter',
    ],
  fortran_args: [
    '-I' + numpy_inc,
    '-I' + f2py_inc,
    '-cpp'
  ],
  install: true,
  subdir: 'drs'
)
